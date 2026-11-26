import csv
import json
import time
from pathlib import Path
import requests

API_URL = "http://localhost:8000/preguntar"
RUTA_PREGUNTAS = Path("datos/preguntas_evaluacion.csv")

resultados = []

with RUTA_PREGUNTAS.open(encoding="utf-8", newline="") as archivo:
    for fila in csv.DictReader(archivo):
        inicio = time.perf_counter()
        try:
            respuesta = requests.post(
                API_URL,
                json={
                    "pregunta": fila["pregunta"],
                    "top_k": 3,
                    "temperatura": 0.2
                },
                timeout=360
            )
            respuesta.raise_for_status()
            datos = respuesta.json()
            resultados.append({
                "id": fila["id"],
                "pregunta": fila["pregunta"],
                "esperada": fila["respuesta_esperada"],
                "generada": datos["respuesta"],
                "tokens_entrada": datos["tokens_entrada"],
                "tokens_salida": datos["tokens_salida"],
                "duracion": time.perf_counter() - inicio
            })
            print(f"Pregunta {fila['id']} completada")
        except requests.RequestException as error:
            print(f"Error en pregunta {fila['id']}: {error}")

Path("datos/resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("Evaluacion terminada")
