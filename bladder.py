from dotenv import load_dotenv
import os
import cv2
import asyncio
import numpy as np
import tensorflow as tf
import edge_tts
import subprocess

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug
from typing import TypedDict
from langgraph.graph import StateGraph, START, END 



load_dotenv()

# =========================
# CONFIGURAÇÕES
# =========================

MODEL_TFLITE_PATH = "ADD-PATH"
VIDEO_PATH = "ADD-PATH"

OUTPUT_VIDEO = "ADD-PATH"
OUTPUT_AUDIO = "ADD-PATH"
VIDEO_FINAL = "ADD-PATH"

CONF_MINIMA = 0.70
ANALISAR_A_CADA_N_FRAMES = 90

# =========================
# LLM - LM STUDIO
# =========================

modelo = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.5
)


prompt_guide_bladder = ChatPromptTemplate.from_messages(
    [
        ("system", """
            Você é uma auxiliar de aquisição de imagens de ultrassom da bexiga em tempo real.

            Sua função é orientar o operador com base em dados de segmentação da bexiga.
            Você não realiza diagnóstico médico.
            Você não inventa achados clínicos.
            Você apenas comenta qualidade da imagem, presença da bexiga, estabilidade da segmentação e orientação de aquisição.

            Responda em no máximo duas frases.
            Use linguagem simples.
            Não utilize markdown, asteriscos, hashtags, listas ou emojis.
        """),
        ("human", "{query}")
    ]
)

# Criação das cadeias para cada consultor
chain_bladder = prompt_guide_bladder | modelo | StrOutputParser()


# =========================
# MODELO TFLITE
# =========================

class BladderTFLite:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

        self.input_shape = self.input_details[0]["shape"]
        
        
        print("Input shape:", self.input_shape) 
        print("Output shape:", self.output_details[0]["shape"])

    def preprocess(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, (224, 360))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_input = np.expand_dims(frame_norm, axis=(0, -1))
        return frame_input

    def predict(self, frame):
        x = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()

        pred = self.interpreter.get_tensor(self.output_index)[0, :, :, 0]
        mask = (pred > 0.5).astype(np.uint8)

        return pred, mask

# =========================
# MÉTRICAS DA BEXIGA
# =========================

def calcular_metricas_bladder(pred, mask):
    area = int(np.sum(mask))

    confidence = float(np.mean(pred[mask == 1])) if area > 0 else 0.0

    if area > 0:
        ys, xs = np.where(mask == 1)
        center_x = int(np.mean(xs))
        center_y = int(np.mean(ys))
    else:
        center_x = None
        center_y = None

    bladder_detected = area > 500 and confidence >= CONF_MINIMA

    return {
        "bladder_detected": bladder_detected,
        "mask_area": area,
        "confidence": round(confidence, 3),
        "center_x": center_x,
        "center_y": center_y
    }

# =========================
# LANGGRAPH STATE
# =========================

class BladderState(TypedDict):
    frame_id: int
    bladder_detected: bool
    mask_area: int
    confidence: float
    center_x: int | None
    center_y: int | None
    query: str
    resposta: str

# =========================
# NODE LLM
# =========================

async def gerar_orientacao_bladder(state: BladderState):
    query = f"""
    Analise os dados abaixo de uma segmentação de bexiga em ultrassom.

    Frame: {state["frame_id"]}
    Bexiga detectada: {state["bladder_detected"]}
    Área da máscara: {state["mask_area"]}
    Confiança: {state["confidence"]}
    Centro X: {state["center_x"]}
    Centro Y: {state["center_y"]}

    Gere uma orientação curta para o operador.
    Não diga que é diagnóstico médico.
    Fale como uma auxiliar de aquisição de imagem.
    Seja objetiva.
    """

    resposta = await chain_bladder.ainvoke({"query": query})

    resposta = (
        resposta.replace("*", "")
        .replace("#", "")
        .replace("-", "")
        .replace("**", "")
        .strip()
    )

    return {"resposta": resposta}

# =========================
# GRAFO
# =========================

workflow = StateGraph(BladderState)

workflow.add_node("gerar_orientacao_bladder", gerar_orientacao_bladder)

workflow.add_edge(START, "gerar_orientacao_bladder")
workflow.add_edge("gerar_orientacao_bladder", END)

graph_bladder = workflow.compile()

# =========================
# TEXTO PARA ÁUDIO
# =========================

async def gerar_audio(texto, output_audio):
    communicate = edge_tts.Communicate(
        texto,
        voice="pt-BR-FranciscaNeural"
    )
    await communicate.save(output_audio)

# =========================
# PROCESSAR VÍDEO
# =========================

async def processar_video():
    bladder_model = BladderTFLite(MODEL_TFLITE_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise Exception("Erro ao abrir vídeo.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # É um código de 4 caracteres que identifica o formato de compressão do vídeo.
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_id = 0
    ultima_resposta = "Iniciando análise da bexiga."

    respostas_audio = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        pred, mask = bladder_model.predict(frame)
        metricas = calcular_metricas_bladder(pred, mask)

        frame_final = frame.copy()

        if frame_id % ANALISAR_A_CADA_N_FRAMES == 0:
            state = {
                "frame_id": frame_id,
                "bladder_detected": metricas["bladder_detected"],
                "mask_area": metricas["mask_area"],
                "confidence": metricas["confidence"],
                "center_x": metricas["center_x"],
                "center_y": metricas["center_y"],
                "query": "",
                "resposta": ""
            }

            result = await graph_bladder.ainvoke(state)
            ultima_resposta = result["resposta"]

            print(f"Frame {frame_id}: {ultima_resposta}")
            respostas_audio.append(ultima_resposta)

        cv2.putText(
            frame_final,
            f"Conf: {metricas['confidence']}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame_final,
            ultima_resposta[:80],
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        writer.write(frame_final)
        frame_id += 1

    cap.release()
    writer.release()

    if respostas_audio:
        texto_audio = ". ".join(respostas_audio)
        await gerar_audio(texto_audio, OUTPUT_AUDIO)
        print("Áudio salvo em:", OUTPUT_AUDIO)
    else:
        print("Nenhuma resposta foi gerada para áudio.")

    print("Vídeo salvo em:", OUTPUT_VIDEO)
    print("Áudio salvo em:", OUTPUT_AUDIO)

    if os.path.exists(OUTPUT_VIDEO) and os.path.exists(OUTPUT_AUDIO):
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", OUTPUT_VIDEO,
            "-i", OUTPUT_AUDIO,
            "-c:v", "copy",
            "-c:a", "aac",
            VIDEO_FINAL
        ], check=True)

        print("Vídeo final salvo em:", VIDEO_FINAL)
    else:
        print("Não foi possível gerar o vídeo final. Vídeo ou áudio não encontrado.")

if __name__ == "__main__":
    asyncio.run(processar_video())
