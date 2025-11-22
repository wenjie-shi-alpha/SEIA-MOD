# SEIA-Mod 3.0 实施行动指南 (Implementation Action Guide)

本指南旨在将 SEIA-Mod 3.0 的设计理念转化为可执行的工程落地手册。指南侧重于**工程实现路径**、**代码架构设计**与**关键技术攻关**。

---

## 1. 项目脚手架与架构 (Project Scaffolding)

### 1.1 目录结构规范
建立清晰的模块化结构，分离数据处理、仿真内核与交互层。

```bash
SEIA-Mod/
├── data/                  # 挂载 S3/MinIO 的本地映射或临时数据
├── docker/                # 基础设施编排 (ArangoDB, MinIO)
├── src/
│   ├── ingestion/         # L1: 数据虚拟化脚本 (Kerchunk, GeoParquet)
│   ├── graph/             # L2: 图谱构建与查询 (ArangoDB Driver)
│   ├── simulation/        # L3: 核心计算层
│   │   ├── meteo_cognition.py # 3.1: 降尺度与事件生成 (Prithvi/SAM)
│   │   ├── mapper.py          # 3.2: 时空映射与子图加载
│   │   ├── agents/            # 3.3: Hybrid Agent 定义
│   │   └── model.py           # 3.3: Mesa 仿真主循环
│   ├── cognition/         # L3: LLM API Client (通用工具)
│   └── server/            # L4: WebSocket 流服务 (FastAPI)
├── frontend/              # L4: 可视化端 (Vue3 + Deck.gl)
├── .env                   # API Keys 与配置
└── requirements.txt
```

### 1.2 基础设施编排 (Docker Compose)
**行动**：编写 `docker-compose.yml` 启动核心依赖。
*   **ArangoDB**: 存储多模态图谱（承灾体、事件、关系）。
*   **MinIO**: 模拟 S3 对象存储（存放气象数据）。
*   **注意**: 本版本移除本地 LLM 推理服务，改用云端 API 以降低部署门槛。需在 `.env` 中配置 `LLM_API_KEY`。

---

## 2. L1: 数据虚拟化流水线 (Data Virtualization)

**核心目标**：实现 PB 级气象数据的“零拷贝”按需读取。

### 2.1 气象数据索引化 (Weather Cube)
**行动**：编写 `src/ingestion/indexer.py`。
1.  **扫描**: 使用 `kerchunk.grib2.scan_grib` 扫描原始 GRIB2 文件。
2.  **聚合**: 使用 `MultiZarrToZarr` 将多个时间步的 JSON 索引合并为一个逻辑 Zarr 组。
3.  **存储**: 将生成的 `.json` 引用文件存入 S3/MinIO，而非转换原始数据。

### 2.2 静态资产列式化 (Vector Assets)
**行动**：编写 `src/ingestion/vectorizer.py`。
1.  **转换**: 将 Shapefile/GeoJSON 转换为 **GeoParquet**。
2.  **清洗**: 使用 **DuckDB** SQL 语句进行空间过滤，仅保留仿真区域数据。

---

## 3. L2: 时空承灾体图谱 (Spatio-Temporal Exposure Graph)

**核心目标**：构建高保真的物理世界映射，解决异构数据的时空对齐与动态关联。

### 3.1 空间索引标准化
**行动**：强制所有空间数据通过 **Uber H3 (Res 7-9)** 进行网格化。
*   **代码规范**：所有 Agent、气象 Grid、统计数据必须包含 `h3_index` 字段作为主键或外键。
*   **层级策略**：
    *   **Res 7 (~1.2km)**: 用于宏观气象场聚合。
    *   **Res 9 (~0.17km)**: 用于精细化承灾体（建筑物、工厂）定位。

### 3.2 图谱 Schema 定义 (ArangoDB)
**行动**：在 ArangoDB 中创建多模态集合，支持“文档+图”混合查询。

*   **Document Collections (实体节点)**:
    *   `Assets`: 静态承灾体。
        ```json
        { "_key": "h3_id_uuid", "type": "factory", "sector": "electronics", "capacity": 100, "elevation": 15.5 }
        ```
    *   `WeatherEvents`: 动态气象事件。
        ```json
        { "_key": "evt_20251122_01", "type": "typhoon", "h3_coverage": ["89283...", ...], "intensity_index": 0.85 }
        ```
    *   `Vulnerability`: 脆弱性曲线（函数定义）。
        ```json
        { "_key": "vul_factory_flood", "curve_type": "sigmoid", "params": { "threshold": 0.5, "slope": 2.0 } }
        ```

*   **Edge Collections (关系边)**:
    *   `SupplyChain`: 供应链拓扑。
        ```json
        { "_from": "Assets/A", "_to": "Assets/B", "type": "logistics", "transport_mode": "road" }
        ```
    *   `Exposure`: 暴露关系（动态生成）。
        ```json
        { "_from": "WeatherEvents/E", "_to": "Assets/A", "impact_potential": 0.7 }
        ```
    *   `HasVulnerability`: 资产与脆弱性曲线的关联。
        ```json
        { "_from": "Assets/A", "_to": "Vulnerability/V" }
        ```

### 3.3 承灾体初始化流水线
**行动**：编写 `src/graph/loader.py`。
1.  **ETL**: 读取 GeoParquet -> 计算 H3 -> 映射属性 -> 批量写入 `Assets` 集合。
2.  **拓扑构建**: 根据路网或供应链数据，批量创建 `SupplyChain` 边。
3.  **脆弱性绑定**: 根据资产类型（如“化工厂”），自动关联对应的 `Vulnerability` 节点。

---

## 4. L3: 核心计算层 (Kernel: Neuro-Symbolic Hybrid Engine)

**核心目标**：实现从“物理降尺度”到“社会仿真”的完整闭环。本层分为三个紧密耦合的子模块。

### 4.1 子模块 3.1: 多尺度气象认知 (Meteorological Cognition)
**功能**：先“看清”（降尺度），再“看懂”（特征提取），最后“表达”（事件生成）。

**行动**：开发 `src/simulation/meteo_cognition.py`。
1.  **AI 降尺度 (Downscaling)**:
    *   集成 **Prithvi WxC** (2.3B 参数模型)。
    *   输入：L1 提供的粗分辨率气象场 (ERA5, ~30km)。
    *   输出：高分辨率气象张量 (2km 级)，捕捉局地极端天气。
2.  **地理特征分割 (Segmentation)**:
    *   集成 **SAM-Geo** 模型。
    *   对高分气象张量进行掩膜分割，识别“台风眼”、“洪涝淹没区”等具体对象。
3.  **语义事件生成 (Event Generation)**:
    *   调用 LLM API (Qwen/DeepSeek)。
    *   将分割后的掩膜统计值（如中心风速、覆盖 H3 列表）转化为结构化 JSON 事件对象。

### 4.2 子模块 3.2: 动态时空映射 (Dynamic Spatio-Temporal Mapper)
**功能**：负责“场景加载”，将物理事件映射为仿真环境的初始状态。

**行动**：开发 `src/simulation/mapper.py`。
1.  **空间索引对齐**:
    *   接收事件 JSON 中的 H3 列表。
    *   执行 ArangoDB AQL 查询，毫秒级锁定受影响的 `Assets` 节点。
2.  **脆弱性上下文加载 (Vulnerability Context Loading)**:
    *   **修正逻辑**: 此阶段**不计算**具体损毁值。
    *   而是将节点对应的脆弱性曲线参数（如 `{"curve_type": "sigmoid", "threshold": 50mm}`）一并加载到内存中，绑定到 Agent 对象上，供后续仿真步使用。
3.  **子图实例化 (Subgraph Instantiation)**:
    *   提取受损节点及其 $N$ 跳邻居（供应链上下游）。
    *   构建轻量级**内存仿真子图**，准备注入 Mesa。

### 4.3 子模块 3.3: 混合智能体仿真 (Hybrid Agent-Based Simulation)
**功能**：在子图上推演社会演化，双脑协同。

**行动**：完善 `src/simulation/model.py` 和 `agents/hybrid_agent.py`。
*   **架构**: **Mesa (容器) + LLM API (决策函数)**。
*   **仿真循环 (The Step Loop)**:
    1.  **物理规则步 (Physics Step)**:
        *   **动态损毁计算**: 读取当前 Tick 的气象值，代入 Agent 的脆弱性曲线函数，计算当前物理损毁（如 `damage_rate = sigmoid(rain_amount)`）。
        *   **守恒计算**: 基于损毁率扣减产能、库存。
    2.  **认知决策步 (Cognition Step)**:
        *   **触发器**: 仅当 Agent 状态触及非线性临界点（如“库存 < 安全线” 且 “物流中断”）时触发。
        *   **API 调用**: 将局部 Context 发送给 LLM API。
        *   **决策执行**: 解析返回的 JSON（如 `{"action": "panic_buy"}`），更新 Agent 状态并向邻居发送消息。
    3.  **宏观统计**:
        *   Mesa `DataCollector` 汇总每步的 GDP 损失与风险扩散指数。

---

## 5. L4: 高频交互与可视化 (Interaction)

**核心目标**：解决百万级 Agent 的实时渲染瓶颈。

### 5.1 二进制流通信
**行动**：开发 `src/server/socket_server.py`。
*   **协议**: 定义简单的二进制协议头。
*   **Agent 状态**: 使用 `numpy.tobytes()` 发送 Agent 位置与状态数组 (Float32)，**严禁**使用 JSON。
*   **气象场数据**:
    *   **优化策略**: 对于百万级气象粒子，**不要**直接传输粒子位置。
    *   **方案**: 后端仅发送**流场纹理 (Vector Field Texture)** 或稀疏的矢量网格数据。
    *   **前端计算**: 前端 Deck.gl/WebGPU 接收流场数据，在 Shader 中进行粒子平流 (Advection) 计算，大幅降低带宽压力。

### 5.2 GPU 直通渲染
**行动**：前端使用 **Deck.gl** 开发。
*   **Agent 图层**: 使用 `ScatterplotLayer` 渲染离散 Agent，数据源为 WebSocket 二进制流。
*   **气象图层**: 使用 `ParticleLayer` (自定义 WebGPU 层)，基于接收到的流场纹理进行 GPU 粒子模拟。

---

## 6. 开发与验证清单 (Checklist)

### 阶段一：数据底座 (Week 1)
- [ ] S3 服务跑通，Kerchunk 索引生成成功。
- [ ] Xarray 能远程读取气象切片，耗时 < 500ms。
- [ ] ArangoDB 成功导入 GeoParquet 转换后的 H3 节点数据。

### 阶段二：仿真内核 (Week 2-3)
- [ ] 集成 Prithvi WxC 模型，跑通气象降尺度流程。
- [ ] 实现 SAM-Geo 分割，能从气象场提取事件掩膜。
- [ ] 完成 ArangoDB 子图提取逻辑，实现“场景加载”。
- [ ] Mesa 混合仿真跑通：物理规则正常，API 决策触发正确。

### 阶段三：全链路可视化 (Week 3)
- [ ] WebSocket 服务端能以 60Hz 推送二进制流。
- [ ] 前端 Deck.gl 能渲染 10万+ 动态粒子不掉帧。

---

此指南摒弃了理论阐述，直接给出了**代码级**的实施路径。请按照目录结构初始化项目，并按阶段清单逐一攻克。
