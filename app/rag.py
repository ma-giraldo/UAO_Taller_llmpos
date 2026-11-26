import os
import json
import time
from pathlib import Path
import numpy as np
import requests
import mlflow

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODELO_EMBEDDING = "nomic-embed-text"
RUTA_DOCUMENTOS = Path("/documentos")
RUTA_INDICE = Path("/datos/indice.json")


# 1. Leer archivos TXT
def leer_documentos():
    documentos = []
    for archivo in sorted(RUTA_DOCUMENTOS.glob("*.txt")):
        documentos.append({
            "fuente": archivo.name,
            "texto": archivo.read_text(encoding="utf-8")
        })
    return documentos


# 2. Dividir textos con solapamiento
def dividir_texto(texto, tamano=500, solape=80):
    if tamano <= solape:
        raise ValueError("El tamaño debe superar el solape")
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        fragmentos.append(texto[inicio:fin])
        if fin == len(texto):
            break
        inicio = fin - solape
    return fragmentos


# 3. Embeddings mediante Ollama
def crear_embedding(texto):
    respuesta = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": MODELO_EMBEDDING, "input": texto},
        timeout=180
    )
    respuesta.raise_for_status()
    return respuesta.json()["embeddings"][0]


# 4. Indexar documentos
def indexar_documentos():
    documentos = leer_documentos()
    indice = []
    for documento in documentos:
        fragmentos = dividir_texto(documento["texto"])
        for numero, fragmento in enumerate(fragmentos):
            indice.append({
                "fuente": documento["fuente"],
                "fragmento": numero + 1,
                "texto": fragmento,
                "embedding": crear_embedding(fragmento)
            })
    RUTA_INDICE.parent.mkdir(parents=True, exist_ok=True)
    RUTA_INDICE.write_text(
        json.dumps(indice, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return {"documentos": len(documentos), "fragmentos": len(indice)}


# 5. Comparar vectores
def similitud_coseno(vector_a, vector_b):
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)
    denominador = np.linalg.norm(a) * np.linalg.norm(b)
    if denominador == 0:
        return 0.0
    return float(np.dot(a, b) / denominador)


# 6. Recuperar los top_k fragmentos
@mlflow.trace(name="recuperar_documentos")
def recuperar_documentos(pregunta, top_k=3):
    inicio = time.perf_counter()
    if not RUTA_INDICE.exists():
        raise FileNotFoundError("Primero debes indexar los documentos")
    indice = json.loads(RUTA_INDICE.read_text(encoding="utf-8"))
    if not indice:
        raise ValueError("El índice está vacío")
    vector_pregunta = crear_embedding(pregunta)
    resultados = []
    for item in indice:
        resultados.append({
            "fuente": item["fuente"],
            "fragmento": item["fragmento"],
            "texto": item["texto"],
            "similitud": similitud_coseno(
                vector_pregunta, item["embedding"]
            )
        })
    resultados.sort(key=lambda x: x["similitud"], reverse=True)
    return {
        "fragmentos": resultados[:top_k],
        "tiempo_recuperacion": time.perf_counter() - inicio
    }
