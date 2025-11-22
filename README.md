# SEIA-Mod

SEIA-Mod 是一个围绕“数据虚拟化 → 时空图谱 → 神经-符号混合仿真 → 高频交互”四层架构构建的工程化方案。它将 PB 级气象/矢量数据的零拷贝访问、ArangoDB 图谱、Mesa+LLM 的双脑仿真以及 FastAPI/WebSocket 的二进制流传输组合在一起，以支撑极端天气对复杂供应链网络的弹性评估。

本仓库包含：

- **L1 数据虚拟化**：Kerchunk GRIB2 索引、GeoParquet 矢量列式化。
- **L2 图谱装载**：H3 网格化、ArangoDB 批量导入、供应链 + 脆弱性边。
- **L3 仿真内核**：Prithvi/SAM-Geo 事件认知、动态场景映射、Mesa + LLM 的混合智能体。
- **L4 交互层**：FastAPI WebSocket 二进制协议，支撑 Deck.gl/WebGPU 前端。
- **端到端测试**：Pytest 覆盖 ingestion/graph/simulation/server 栈。

---

## 目录结构

```
SEIA-Mod/
├── README.md              # 本文件
├── schedule.md            # 架构推进计划与里程碑
├── guide.md               # 设计理念与实现指南
├── idea.md                # 架构理念 / 设计哲学
├── requirements.txt       # Python 依赖
├── .env                   # 需填写的运行时配置（未入库）
├── docker/
│   └── docker-compose.yml # ArangoDB + MinIO 基座
├── data/                  # 数据挂载目录(含 .gitkeep)
├── frontend/
│   └── README.md          # Vue + Deck.gl 客户端规划
├── src/
│   ├── ingestion/         # L1: WeatherCube & Vectorizer
│   ├── graph/             # L2: Exposure Graph Loader
│   ├── simulation/        # L3: Meteo cognition + mapper + agents + model
│   ├── cognition/         # LLM API 客户端
│   └── server/            # L4: FastAPI WebSocket 流服务
└── tests/                 # Pytest 覆盖 ingestion/graph/simulation/server
```

---

## 核心模块

### L1 数据虚拟化 (`src/ingestion/`)

- `indexer.py` (`WeatherCubeIndexer`)：使用 Kerchunk 的 `scan_grib` + `MultiZarrToZarr` 组合生成 GRIB2 引用文件（Reference JSON），并可选上传至 MinIO/S3。
- `vectorizer.py` (`VectorAssetVectorizer`)：用 GeoPandas + DuckDB 读取 Shapefile/GeoJSON，按 SQL 过滤仿真区域后输出为 GeoParquet，便于下游列式扫描。

### L2 时空承灾图谱 (`src/graph/`)

- `loader.py` (`ExposureGraphLoader`)：读取 GeoParquet，转换 H3 指标，批量写入 ArangoDB `Assets` 集合，并构建 `SupplyChain`/`HasVulnerability` 边；提供 `impacted_assets_aql` 供上层查询。

### L3 混合仿真引擎 (`src/simulation/`)

1. `meteo_cognition.py`：定义 `MeteoCognitionPipeline`，将可插拔的降尺度器（Prithvi WxC）、分割器（SAM-Geo）与 LLM 客户端串联生成结构化事件 (`EventObservation`)。
2. `mapper.py`：`DynamicMapper` + `ArangoGraphBackend` 根据事件的 H3 覆盖区，拉取资产、绑定脆弱性上下文，并输出带 N-hop 供应链的仿真子图。
3. `agents/hybrid_agent.py`：实现 `HybridAgent`，包含脆弱性曲线计算、产能/库存守恒逻辑、基于 LLM 的认知触发。
4. `model.py`：`HybridSimulationModel` 负责载入子图、执行 tick 循环、汇总宏观指标，支持自定义 LLM 客户端与多步骤运行。

### L4 高频交互 (`src/server/`)

- `socket_server.py`：FastAPI 应用，包含
  - `/ingest` REST 入口：接收 Agent 批次，编码为自定义 SEIA Header + Float32 Payload。
  - `/ws/agents` WebSocket：推送二进制帧（支持 60Hz 队列），供 Deck.gl/WebGPU 客户端消费。
- `AgentStateBroadcaster` 抽象了队列和编码逻辑，确保多消费者可共享同一帧。

### 测试 (`tests/`)

- `test_ingestion.py`：验证 Kerchunk 索引流程与 GeoParquet 向量化。
- `test_graph_loader.py`：通过 Fake Arango 客户端校验资产/边导入与 AQL 片段。
- `test_simulation.py`：对气象认知、动态映射、混合仿真进行端到端 stub 测试。
- `test_server.py`：检查二进制编码头信息与 WebSocket 推送一致性。

---

## 快速开始

### 1. 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ `requirements.txt` 包含 Kerchunk、GeoPandas、ArangoDB 驱动等重量级依赖，建议在 Python 3.11 / 3.12 环境中使用虚拟环境安装。

### 2. 配置 `.env`

复制 `.env` 模板，填入本地/云端凭据：

```
ARANGODB_URL=http://localhost:8529
ARANGODB_DB=seia_mod
ARANGODB_USER=root
ARANGODB_PASSWORD=...
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
LLM_API_KEY=...
```

### 3. 启动基础设施

```bash
cd docker
docker compose up -d
# 验证
docker compose ps
```

该 `docker-compose.yml` 会启动：
- **ArangoDB 3.11**（暴露 8529）
- **MinIO**（暴露 9000/9001）

### 4. 运行测试

```bash
.venv/bin/pytest
```

当前 8 个测试用例覆盖 L1–L4 核心路径，建议在提交前保持全绿。

---

## 典型工作流

1. **数据虚拟化 (L1)**
   ```python
   from pathlib import Path
   from src.ingestion.indexer import WeatherCubeIndexer, WeatherCubeIndexerConfig

   config = WeatherCubeIndexerConfig(
       source_glob="/data/grib/**/*.grib2",
       output_dir=Path("data/indexes"),
       s3_bucket="seia-weather",
       object_store_options={"client_kwargs": {"endpoint_url": "http://localhost:9000"}},
   )
   WeatherCubeIndexer(config).run()
   ```
   生成的 `consolidated.json` 可被 Xarray 远程读取。

2. **图谱导入 (L2)**
   ```python
   from pathlib import Path
   from src.graph.loader import ExposureGraphLoader, GraphLoaderConfig

   loader = ExposureGraphLoader(
       GraphLoaderConfig(
           geo_parquet_path=Path("data/assets.parquet"),
           arangodb_url="http://localhost:8529",
           database="seia_mod",
           username="root",
           password="change-me",
       )
   )
   loader.run()
   ```

3. **仿真准备与执行 (L3)**
   ```python
   from src.simulation.mapper import DynamicMapper, MapperConfig
   from src.simulation.model import HybridSimulationModel, SimulationConfig

   mapper = DynamicMapper(
       MapperConfig(
           arangodb_url="http://localhost:8529",
           database="seia_mod",
           username="root",
           password="change-me",
           neighborhood_hops=2,
       )
   )
   subgraph = mapper.run({"h3_coverage": ["892830828cbffff"]})

   simulation = HybridSimulationModel(SimulationConfig())
   simulation.load_subgraph(subgraph)
   metrics = simulation.step({"hazard_index": 0.8, "demand_multiplier": 1.5}, llm_client=...)
   ```

4. **实时交互 (L4)**
   - 启动 FastAPI WebSocket 服务器（例如 `uvicorn src.server.socket_server:app --reload`）。
   - 前端（Deck.gl/Vue）通过 `ws://<host>/ws/agents` 获取 Float32 二进制流并渲染。

---

## 前端规划（`frontend/`）

`frontend/README.md` 描述了 Vue 3 + Vite + Deck.gl 的搭建路线：

1. 初始化 Vite Vue 项目。
2. 接入 WebSocket 二进制流，使用 TypedArray 解析 `SEIA` 头。
3. 利用 Deck.gl 的 `ScatterplotLayer` + 自定义 WebGPU `ParticleLayer` 实现百万级 Agent + 气象粒子渲染。

> 当前尚未创建实际前端代码，可根据该说明启动实现。

---

## 未来路线 (节选自 `schedule.md`)

1. **前端可视化**：完成 Vite + Deck.gl 客户端、二进制协议对接。
2. **验证与部署**：运行 Docker 基座、完善 CI/CD、性能压测。

更多任务详情与完成记录见 `schedule.md`。

---

## 参考 & 致谢

- [Kerchunk](https://fsspec.github.io/kerchunk/)
- [ArangoDB](https://www.arangodb.com/)
- [Mesa](https://mesa.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Deck.gl](https://deck.gl/)

如果你计划在此框架上扩展更多模型或 UI，也欢迎更新 `schedule.md` 以同步里程碑。 Happy hacking! 🎯
