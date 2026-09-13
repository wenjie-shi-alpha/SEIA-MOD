# SEIA-MOD

面向极端天气影响评估的混合仿真框架。将气象事件映射到承灾体与供应链，用物理规则和 LLM 行为决策分析直接损毁及下游影响。

## 核心设计

- **时空数据与图谱**：连接气象场、地理资产与脆弱曲线，按事件范围提取受影响的供应链子图。
- **物理与语言决策**：规则推进损毁、产能和库存状态，在压力阈值触发智能体的行为响应。
- **影响传导**：计算资产直接受损与下游供给缺口，输出节点及区域指标，提供状态服务接口。

## 快速开始

建议使用 Python 3.11 / 3.12：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_simulation.py
```

以上测试使用合成数据和模拟客户端。真实案例需要另外准备气象与资产数据，并配置所用数据库和模型服务，详见[运行指南](docs/usage.md)。

## 文档

- [运行与案例](docs/usage.md)
- [架构设计](guide.md)
- [外部资产包接口](docs/external_asset_pipeline.md)
- [脆弱曲线方法](docs/vulnerability_curve_methodology.md)

仓库以数据、图谱与仿真后端为主；可视化前端的设计见 [`frontend/README.md`](frontend/README.md)。
