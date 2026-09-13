# 运行与案例

[返回首页](../README.md)

## 环境

在仓库根目录创建 Python 3.11 / 3.12 虚拟环境，安装 `requirements.txt`。
该文件包含数据处理、地理空间、图谱与测试依赖。模型相关扩展另见 `requirements-ml.txt`。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_simulation.py
```

仿真测试使用合成气象数据和模拟客户端。完整测试入口为 `python -m pytest`。

## 配置存储服务

使用 ArangoDB / MinIO 时，在本地 `.env` 配置 `ARANGODB_PASSWORD`、`MINIO_ACCESS_KEY` 和 `MINIO_SECRET_KEY`，再启动仓库提供的开发服务：

```bash
docker compose --env-file .env -f docker/docker-compose.yml up -d
```

服务定义见 [docker-compose.yml](../docker/docker-compose.yml)。
业务侧数据库连接、对象存储和模型客户端配置应与实际运行环境一致；模型端点与凭据通过 [LLMClientConfig](../src/cognition/llm_client.py) 传入。

## 准备资产并运行案例

资产层采用带 H3 索引与脆弱性标签的 GeoParquet。可以在本仓生成，也可通过外部资产包接入：

- [资产包结构及加载](external_asset_pipeline.md)
- [脆弱曲线与版本](vulnerability_curve_methodology.md)

[京津冀高温脚本](../scripts/run_heatwave_impact.py) 使用以下两个输入：

```text
data/vectorized/jjj_power_assets.parquet
resources/vulnerability/jjj_power_vulnerabilities.json
```

输入准备完成后，在仓库根目录运行：

```bash
PYTHONPATH=. python scripts/run_heatwave_impact.py
```

该例将设定的高温场景映射为资产损毁与供应缺口，是情景模拟，不代表已验证的真实灾害损失。
其他模型和气象认知示例位于 [examples/](../examples/)。

## 状态接口

```bash
uvicorn src.server.socket_server:app --reload
```

`/ingest` 接收智能体状态，`/ws/agents` 提供二进制状态流。
前端仍是设计说明，见 [frontend/README.md](../frontend/README.md)；不应将其规划规模视为已测性能。
