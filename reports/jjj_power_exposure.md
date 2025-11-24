# 京津冀电力承灾体承灾体层构建报告

> 产业选择：电力生产（燃煤热电 + 可再生补充）；数据资产：WRI Global Power Plant Database v1.3、Natural Earth Admin-1；脚本：`scripts/build_jjj_power_assets.py`

---

## 1. 与指南/排期的对齐
- 参考 `guide.md` 中 L2 要求：所有实体统一 H3（本层使用 Res-9，共 222 个六边形覆盖京津冀电力骨干节点），并保留 `type/sector/criticality` 供 L3-L4 使用。
- `schedule.md` 的“已完成内容”中已有 `ExposureGraphLoader` 与 GeoParquet 流程，本次构建直接产出与其兼容的 `data/vectorized/jjj_power_assets.parquet`，可作为后续 L3 映射的默认承灾体底座。
- 电力行业选择理由：京津冀地区承压“保电/保热”双重任务，且燃煤热电厂对暴雨、洪水、海潮等气象事件高度敏感，符合指南所述“构建高保真物理映射”的优先级。

---

## 2. 数据来源与生成流程
| 步骤 | 内容 | 说明 |
|------|------|------|
| 数据下载 | `global_power_plant_database.csv`（WRI，CC-BY 4.0）<br>`ne_10m_admin_1_states_provinces`（Natural Earth，Public Domain） | 存放在 `data/external/`；脚本自动检查是否存在 |
| 空间过滤 | 利用 Natural Earth 的 Admin-1 边界筛选北京/天津/河北的 229 座电厂 | 精度高于简单 bbox，避免落入山西/山东 |
| 属性增强 | 计算 `fuel_group`、`segment(thermal/renewable)`、`criticality_score (1-5)`、`resilience_tier`、`impact_buffer_km`、`exposure_profile` | 与指南第 3.3 节的“脆弱性绑定”保持一致，可直接映射 Vulnerability 曲线 |
| 构建 H3 | 所有坐标以 WGS84 → H3-Res9，字段 `h3_index` | 满足“Agent/统计数据共享同一 H3 主键”的约束 |
| 生成负载节点 | 依照国家电网分区人工设定 6 个 `grid_node`（北京核心、天津城区/港口、河北北/南/沿海），`supplier_ids` 自动汇总其 fan-in | 这样 `SupplyChain` 边即可描述“发电 → 负荷中心”的拓扑 |
| 脆弱性曲线 | `resources/vulnerability/jjj_power_vulnerabilities.json` 定义 6 组曲线（潮涌 CHP、热浪 CHP、风电、光伏、内陆基准、网荷节点） | 与 `exposure_profile → vulnerability_type` 映射一一对应 |
| 输出 | GeoJSON + GeoParquet + Vulnerability JSON | `data/vector/jjj_power_assets.geojson`、`data/vectorized/jjj_power_assets.parquet`、`resources/vulnerability/jjj_power_vulnerabilities.json` |

复现命令：

```bash
PYTHONPATH=. .venv/bin/python scripts/build_jjj_power_assets.py
PYTHONPATH=. .venv/bin/python scripts/seed_vulnerabilities.py --password "$ARANGO_PW"
```

脚本输出的关键信息（2024-11-23）：
- 229 座电厂 + 6 个网荷节点，总装机 **67.16 GW**。
- 燃料结构：煤 53、气 16、核 1、水 3、光伏 70、风 86。
- 省份分布：河北 199、北京 18、天津 12。

---

## 3. 承灾体 Schema（与 ArangoDB 对齐）

- **实体类型**
  - `power_plant`：真实发电厂 POI，字段 `_key`（来自 `gppd_idnr`）、`fuel_group`、`capacity_mw`、`criticality_score`、`resilience_tier`、`exposure_profile`、`vulnerability_type`、`h3_index` 等。
  - `grid_node`：人工构建的区域负荷中心，作为 `SupplyChain` 的汇点，字段 `supplier_ids` 会列出所有上游电厂 `_key`，并统一 `vulnerability_type = vul_grid_cluster`。
- **关键字段**
  - `grid_node_key`：用于衔接 `grid_node` 与发电厂的分区标签（Beijing core / Tianjin urban / Hebei north/coastal/south）。
  - `impact_buffer_km`：在 L3 `DynamicMapper` 中决定需要加载的 H3 扩散半径。
  - `exposure_profile & vulnerability_type`：`scripts/build_jjj_power_assets.py` 会直接写出 `vulnerability_type`，与 `resources/vulnerability/jjj_power_vulnerabilities.json` 的 `_key` 对应，便于 `HasVulnerability` 边自动建立。
  - `data_source`：WRI GPPD v1.3、Natural Earth、公网规划资料，方便溯源。

---

## 4. 图谱构建与装载
1. **GeoParquet → ArangoDB**

```python
from pathlib import Path
from src.graph.loader import ExposureGraphLoader, GraphLoaderConfig

config = GraphLoaderConfig(
    geo_parquet_path=Path("data/vectorized/jjj_power_assets.parquet"),
    arangodb_url="http://localhost:8529",
    database="seia_mod",
    username="root",
    password="***",
    h3_resolution=9,
    supplier_field="supplier_ids",
    vulnerability_field="vulnerability_type",
)

loader = ExposureGraphLoader(config)
loader.run()
```

2. **边构建逻辑**
   - `power_plant` 的 `supplier_ids` 为空 → 不会生成上游边。
   - `grid_node` 的 `supplier_ids` 自动展开，例如 `GRID_HEBEI_NORTH` fan-in 115 条发电厂 `_key`，引擎会写入 115 条 `SupplyChain` 边，表达“某发电厂向唐山/曹妃甸重工负荷区供电”的逻辑。

3. **影响子图查询**

```aql
// 根据 H3 查询受影响承灾体
FOR asset IN Assets
  FILTER asset.h3_index IN ["892a92da9dbffff", "892a92da9c7ffff"]
  RETURN { _key: asset._key, capacity: asset.capacity_mw, profile: asset.exposure_profile }
```

---

## 5. 结果洞察
- **容量结构**：`thermal` 69 座 / 57.6 GW，`renewable` 160 座 / 9.6 GW，匹配“燃煤兜底 + 风光补链”的区域现实。
- **Top-5 发电站**（均为高危洪水/高温负荷点）：

| 站点 | 省份 | 主燃料 | 装机 (MW) | 负荷分区 |
|------|------|--------|-----------|----------|
| Tianjin Beijiang power station | 天津 | Coal | 3000 | GRID_TIANJIN_PORT |
| Zhangjiakou power station | 河北 | Coal | 2560 | GRID_HEBEI_NORTH |
| Huaneng Shangan power station | 河北 | Coal | 2540 | GRID_HEBEI_SOUTH |
| Dingzhou power station | 河北 | Coal | 2520 | GRID_HEBEI_SOUTH |
| Xibaipo power station | 河北 | Coal | 2400 | GRID_HEBEI_SOUTH |

- **H3 覆盖**：222 个 Res-9 单元；可直接喂给 L3 `DynamicMapper` 进行事件-H3 交集。
- **负荷节点 fan-in**：北京核心(18)、天津城区(8)、天津港(4)、河北北部(115)、河北沿海(13)、河北南部(71)。这些数值可作为 L4 可视化时的节点大小/颜色，辅助突出冗余薄弱点（如天津港只有 4 条上游链路，极易受风暴潮影响）。

---

## 6. 下一步建议
1. **脆弱性曲线绑定**：`scripts/seed_vulnerabilities.py` 先写入 `Vulnerability/*`，随后 `ExposureGraphLoader` 会根据 `vulnerability_type` 自动产生 `HasVulnerability` 边。
2. **场景加载校验**：调用 `DynamicMapper`，以 2023-08-12 暴雨事件的 H3 掩膜试算，验证燃煤/风电混合节点的批量加载耗时（指南要求 ms 级）。
3. **风险可视化**：前端 Deck.gl 可直接消费 `grid_node` fan-in 统计，叠加事件热力层，符合 `schedule.md` 的“前端可视化”下一步。

项目现已具备“产业 POI → H3 → Arango 承灾体图谱”的闭环，可在此基础上扩展燃气、输油管道等其他关键行业。
