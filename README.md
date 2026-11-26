# Laboratorio LLMOps: RAG local con Ollama

Plataforma local de inteligencia artificial generativa para el caso TecnoFábrica, con recuperación aumentada por generación (RAG), observabilidad de trazas con MLflow y métricas con Prometheus y Grafana.

## Arquitectura

| Servicio   | Función                                            | Puerto |
|------------|-----------------------------------------------------|--------|
| ollama     | Generación de texto y embeddings en local            | 11434  |
| api        | API FastAPI (`/indexar`, `/preguntar`, `/metrics`)   | 8000   |
| mlflow     | Trazabilidad de las consultas                        | 5000   |
| prometheus | Recolección de métricas                              | 9090   |
| grafana    | Dashboards sobre las métricas                        | 3000   |

## Requisitos

- Windows 10/11 con WSL2 y Docker Desktop (backend WSL2)
- Conexión a internet para descargar imágenes y modelos

## Puesta en marcha

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama pull nomic-embed-text
curl -X POST http://localhost:8000/indexar
```

## Endpoints

- Swagger: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana (admin/admin en el primer ingreso): http://localhost:3000

Ejemplo de consulta:

```bash
curl -X POST http://localhost:8000/preguntar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Cada cuánto se realiza el mantenimiento preventivo?", "top_k": 3, "temperatura": 0.2}'
```

## Dashboard de Grafana

Importar `grafana/dashboard-monitoreo-llmops.json` desde Dashboards → New → Import.

## Evaluación

```bash
python app/evaluacion.py
```

Compara las respuestas generadas contra `datos/preguntas_evaluacion.csv` y guarda los resultados en `datos/resultados.json`.
