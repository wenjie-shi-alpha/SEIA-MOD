# Setup and example scenarios

[Back to the project](../README.md)

## Environment

Create a Python 3.11 or 3.12 environment in the repository root. `requirements.txt` includes data, geospatial, graph, and test dependencies. Optional model dependencies are listed in `requirements-ml.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_simulation.py
```

Simulation tests use synthetic weather data and stub clients. Run the full suite with `python -m pytest`.

## Storage services

For ArangoDB and MinIO, configure `ARANGODB_PASSWORD`, `MINIO_ACCESS_KEY`, and `MINIO_SECRET_KEY` in a local `.env`, then start the development services:

```bash
docker compose --env-file .env -f docker/docker-compose.yml up -d
```

Service definitions are in [docker-compose.yml](../docker/docker-compose.yml). Application connection settings must match the actual environment. Pass model endpoints and credentials through [LLMClientConfig](../src/cognition/llm_client.py).

## Assets and scenarios

The asset layer uses GeoParquet with H3 indices and vulnerability labels. Build it locally or consume external bundles:

- [Bundle structure and loading](external_asset_pipeline.md) (Chinese)
- [Vulnerability curves and versions](vulnerability_curve_methodology.md) (Chinese)

The [Jing-Jin-Ji heatwave script](../scripts/run_heatwave_impact.py) expects:

```text
data/vectorized/jjj_power_assets.parquet
resources/vulnerability/jjj_power_vulnerabilities.json
```

Once those inputs are prepared, run from the repository root:

```bash
PYTHONPATH=. python scripts/run_heatwave_impact.py
```

This maps a specified heatwave scenario to asset damage and supply shortfalls. It is a scenario simulation, not a validated estimate of historical disaster losses. Other model and weather-cognition examples are in [examples/](../examples/).

## State interface

```bash
uvicorn src.server.socket_server:app --reload
```

`/ingest` accepts agent states and `/ws/agents` streams binary state frames. The frontend remains a design proposal; see [frontend/README.md](../frontend/README.md) (Chinese). Planned scale is not measured performance.
