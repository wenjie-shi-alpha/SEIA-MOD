# SEIA-MOD

A hybrid simulation framework for extreme-weather impact assessment. It maps weather events onto exposed assets and supply chains, combining physical rules with LLM agent decisions to study direct damage and downstream effects.

## Core design

- **Spatial data and graphs:** connect weather fields, assets, and vulnerability curves to extract affected supply-chain subgraphs.
- **Physical and behavioral dynamics:** update damage, capacity, and inventory through rules, triggering agent responses when stress thresholds are reached.
- **Impact propagation:** estimate direct asset damage and downstream supply shortfalls, with node-level and regional outputs.

## Quick start

Python 3.11 or 3.12 is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_simulation.py
```

These tests use synthetic data and stub clients. Real cases require weather and asset data plus the relevant database and model services. See the [run guide](docs/usage.md).

## Documentation

- [Setup and example scenarios](docs/usage.md)
- [Architecture](guide.md) (Chinese)
- [External asset bundles](docs/external_asset_pipeline.md) (Chinese)
- [Vulnerability curve methodology](docs/vulnerability_curve_methodology.md) (Chinese)

The repository focuses on the data, graph, and simulation backend. Frontend plans are described in [`frontend/README.md`](frontend/README.md) (Chinese).
