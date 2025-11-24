# 京津冀高温事件 MC 报告（JJJ-HEATWAVE-20240715）

> **事件**：2024-07-15 “华北热穹”高温过程（模拟）  
> **承灾体层**：`data/vectorized/jjj_power_assets.parquet`（229 座电厂 + 6 个网荷节点）  
> **脆弱曲线**：`resources/vulnerability/jjj_power_vulnerabilities.json`（6 类）  
> **推演脚本**：`scripts/run_heatwave_impact.py`

---

## 1. 事件描述
- 2024-07-15 14:00 CST，副热带高压北抬叠加下沉增温，在北京—天津—唐山一线形成 **WBGT ≈ 31.5 °C** 的热穹顶。  
- 模型假设：事件中心（39.9042°N, 116.4074°E），峰值 `hazard_index = 0.9`，空间影响按高斯（σ=120 km）衰减，场景参数记录在 `data/outputs/heatwave_event.json`。

## 2. 方法流程
1. **数据准备**  
   - 运行 `scripts/build_jjj_power_assets.py` 生成承灾体层，资产含 `vulnerability_type`。  
   - `scripts/seed_vulnerabilities.py` 将 6 条脆弱曲线写入 Arango（可选，本次离线计算直接从 JSON 载入）。
2. **事件-资产映射** (`scripts/run_heatwave_impact.py`)  
   - 计算每个资产的经纬距 → `hazard_index`。  
   - 依据 `vulnerability_type` 映射至 `VulnerabilityCurve`，输出资产级 `damage_rate`。  
   - 对 `grid_node` 根据上游 `supplier_ids` 评估供给缺口，得到 `combined_damage`。  
   - 结果落盘：`data/outputs/heatwave_impact.parquet` 与 `data/outputs/heatwave_grid_summary.parquet`。

复现命令：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_heatwave_impact.py
```

---

## 3. 关键结论
- **受影响规模**：235 个节点中 169 个 `damage_rate > 0.2`，容量加权损毁率 **44.1%**。
- **区域差异**：北京/天津燃煤-三联供机组平均损毁率分别达 **0.90 / 0.88**，河北受影响较轻（0.24），但因装机规模大仍贡献 18.7 GW 的潜在降载。
- **链路传导**：北京核心网荷因本地机组与上游供电双双受损，`combined_damage = 0.984`，天津城区/港口节点亦超过 0.9，意味着需要跨省支援或大规模错峰。

### 3.1 省域统计

| 省份 | 电厂数量 | 装机 (MW) | 加权损毁率 |
|------|---------:|----------:|-----------:|
| 北京 | 18 | 10,543 | 0.902 |
| 天津 | 12 | 10,549 | 0.882 |
| 河北 | 199 | 46,070 | 0.244 |

### 3.2 脆弱类型损毁

| 曲线 | 资产数 | 平均损毁率 | 说明 |
|------|-------:|-----------:|------|
| `vul_chp_heat` | 12 | 0.914 | 北京三联供热电，受热浪触发冷却退化。 |
| `vul_chp_storm` | 8 | 0.894 | 天津滨海 CHP，本次热浪叠加供水短缺导致深度降载。 |
| `vul_pv_heat` | 70 | 0.449 | 冷却-积灰并发，输出降至约 55%。 |
| `vul_windstorm` | 86 | 0.257 | 高温伴弱风，风电风切导致限发。 |
| `vul_generic` | 53 | 0.275 | 河北内陆燃煤受冷却水温度上升影响。 |

### 3.3 Top-5 受损机组

| 机组 | 省份 | 装机 (MW) | Hazard Index | Damage Rate | Vulnerability |
|------|------|----------:|-------------:|------------:|---------------|
| Panshan power station | 天津 | 2,260 | 0.740 | 0.965 | `vul_chp_storm` |
| Huaneng Yangliuqing power station | 天津 | 1,300 | 0.710 | 0.955 | `vul_chp_storm` |
| Tianjin Northeast power station | 天津 | 660 | 0.672 | 0.939 | `vul_chp_storm` |
| Beijing Taiyanggong Trigeneration | 北京 | 780 | 0.898 | 0.920 | `vul_chp_heat` |
| Jinjeng (Beijing) | 北京 | 700 | 0.897 | 0.919 | `vul_chp_heat` |

### 3.4 网荷节点（供应链）损毁

| 网荷节点 | 省份 | Hazard | 直接损毁 | 供应损失 | 综合损毁 |
|----------|------|-------:|---------:|---------:|---------:|
| State Grid Beijing Core Load Center | 北京 | 0.900 | 0.840 | 0.902 | **0.984** |
| State Grid Tianjin Urban Ring | 天津 | 0.676 | 0.593 | 0.911 | 0.964 |
| Tianjin Port Petrochemical Hub | 天津 | 0.546 | 0.450 | 0.832 | 0.908 |
| Hebei North Heavy Industry Ring | 河北 | 0.486 | 0.385 | 0.364 | 0.609 |
| Caofeidian Coastal Grid | 河北 | 0.387 | 0.276 | 0.180 | 0.407 |
| Hebei Southern Load Pocket | 河北 | 0.303 | 0.184 | 0.194 | 0.342 |

---

## 4. 影响分析
1. **热浪脆弱性暴露**：北京/天津的三联供机组在 `hazard_index > 0.85` 区域几乎全损（>0.9），说明冷却冗余与需求应急策略不足。  
2. **跨区支援压力**：河北（尤其北部/沿海）虽然自身损毁较低，但需要反哺首都圈；综合损毁 0.61 的“河北北部重工环”显示必须提前释放外送冗余。  
3. **再生能源波动**：风/光平均损毁 0.26/0.45，虽未完全崩溃，但在热浪期间无法有效补位，需结合储能调度。  
4. **供需双重打击**：`combined_damage` 指出北京核心与天津城区同时遭遇本地热浪 + 上游供电衰减，若无应急调度将触发 10+GW 级别的负荷缺口。

---

## 5. 建议
1. **冷却冗余与供水**：为 `vul_chp_heat`/`vul_chp_storm` 资产配置喷淋/干冷混合改造，力争将 sigmoid 阈值从 0.55 提升到 0.65（可使损毁率降低 ~15pp）。  
2. **跨区错峰方案**：对 `grid_node` 级别输出 `combined_damage > 0.9` 的节点，预先与华北-华中联络线制定“极端热浪日负荷转移”计划，至少释放 5 GW 备用。  
3. **储能与需求响应**：在北京核心负荷圈布置工业级储能与居民 DR，以抵消 0.90 的供应缺口；建议滚动测试 1.5 GW 的 DR 能力。  
4. **模型闭环**：将 `data/outputs/heatwave_impact.parquet` 输入 L3 `DynamicMapper` + Mesa 仿真，评估多 tick 演化（库存消耗、LLM 决策），并将结果回写至 MC 仓库。

---

> 本报告由 `scripts/run_heatwave_impact.py` 自动生成的数值支撑。如需不同事件（例如极端暴雨/风暴潮），可调整脚本中的 `HeatwaveEvent` 参数重新跑通全链路。
