# 外部承灾体数据管线策略

为了让 SEIA-Mod 主仓专注于 L1-L4 框架，本方案将“承灾体数据采集/清洗/构建”置于独立项目（下称 **Asset Hub**）中运行。本仓则仅消费 Asset Hub 输出的“资产包 (Asset Bundle)”并在 ingestion/graph 层留出标准化接口，确保双方松耦合。

## 1. 职责拆分

| 模块 | 负责方 | 主要工作 |
| --- | --- | --- |
| Asset Hub（外部项目） | 数据工程 / 爬虫团队 | 网络爬虫、API 抓取、遥感下载、质量控、GeoParquet & 脆弱性 JSON 产出、版本化、上传对象存储 |
| SEIA-Mod（本仓） | 仿真/建模团队 | 维护接口与 schema、从 Asset Hub 拉取资产包、写入 ArangoDB、驱动仿真与前端 |

双方通过 **Asset Bundle Manifest**（JSON）和一组带校验的资产文件交换数据。本仓不关心 Asset Hub 内部实现，只要遵守 manifest 协议即可。

## 2. 资产包协议

### 2.1 Manifest 结构
Asset Hub 在对象存储或 Git 仓发布 `manifest.json`，结构见 `resources/asset_bundles/manifest.example.json`：

```json
{
  "provider": "external_asset_hub",
  "bundles": [
    {
      "bundle_id": "jjj_power_assets",
      "version": "v20240218",
      "region": "jingjinji",
      "asset_types": ["thermal_power", "grid_node"],
      "hazard_types": ["heatwave", "storm_surge"],
      "asset_count": 1284,
      "geo_parquet": "s3://.../assets.parquet",
      "vulnerability_profiles": "s3://.../vulnerability.json",
      "checksum": "sha256:0d6d1ad5...",
      "metadata": {
        "source_bundle": "catalog://jjj_heatwave_2024",
        "prepared_by": "external-team"
      }
    }
  ]
}
```

必填字段：
- `bundle_id`：资产包唯一 ID（region + hazard 维度）
- `version`：`vYYYYMMDD`，与外部仓 CI 对齐
- `geo_parquet`：资产几何 + 属性（需含 `asset_type`、`industry`、`exposure_profile`、`vulnerability_type`）
- `checksum`：sha256 或 md5，供 SEIA-Mod 校验

可选字段：
- `vulnerability_profiles`：与资产包配套的 JSON（可直接放入 `resources/vulnerability/`）
- `metadata`：引用原始数据集、许可、处理脚本 commit、联系人

### 2.2 文件约定
- **GeoParquet**：必须遵守 `VectorAssetVectorizer` schema；Asset Hub 负责在外部项目完成拼接/标准化。
- **Vulnerability JSON**：字段同 `docs/vulnerability_curve_methodology.md` 中的 schema，可拆分按行业。
- 文件可托管于 OSS/S3/MinIO，也可以是内网路径。SEIA-Mod 仅解析 manifest 中的 URI，并根据配置同步到本地。

## 3. 本仓接口设计

### 3.1 Provider 抽象
- 新增 `src/ingestion/bundle_interface.py`，定义：
  - `AssetBundleManifest` / `AssetBundle` 数据类：描述 manifest 与实际文件。
  - `AssetBundleProvider` 协议：暴露 `list_bundles(region)` 与 `fetch_bundle(manifest)`。
  - `LocalBundleProvider`：默认实现，读取本地 `manifest.json` 并解析相对路径。
- 外部项目若提供 API，可在独立仓实现 `HttpBundleProvider`、`S3BundleProvider` 等，按需注入。

### 3.2 配置形态
在 `.env` 中保留占位项（由用户手动配置）：
```
ASSET_BUNDLE_MANIFEST=resources/asset_bundles/manifest.example.json
ASSET_BUNDLE_ROOT=/mnt/asset_bundles   # 同步后的实际文件目录
```

`scripts/ingestion/pull_asset_bundle.py`（后续待实现）可读取上述变量，调用 Provider：
```python
provider = LocalBundleProvider(Path(env("ASSET_BUNDLE_MANIFEST")), Path(env("ASSET_BUNDLE_ROOT")))
manifest = provider.list_bundles(region="jingjinji")[0]
bundle = provider.fetch_bundle(manifest)
ExposureGraphLoader(
    GraphLoaderConfig(
        geo_parquet_path=bundle.geo_parquet,
        ...
    )
).run()
```

### 3.3 交付流程
1. Asset Hub 产出新资产包 → 上传 GeoParquet/JSON → 更新 manifest。
2. Asset Hub 触发对象存储同步或 rsync 到本仓可访问的目录。
3. 本仓开发者运行 `make asset-bundle`（后续在 `Makefile` 中补充），执行：
   - 下载/同步资产包文件
   - 校验 checksum
   - 使用 Provider 解析 manifest
   - 调用 `ExposureGraphLoader` 完成 ArangoDB 导入

## 4. 外部项目建议结构

```
asset-hub/
├── crawler/               # 各数据源爬虫 & API 客户端
├── normalizer/            # 投影/清洗/标准化脚本
├── bundler/               # 生成 GeoParquet + vulnerability JSON
├── publisher/             # 上传对象存储、更新 manifest
└── manifests/             # 版本化的 manifest.json（推送到 Git 或 OSS）
```

- 产出物：`artifacts/<bundle_id>/<version>/assets.parquet`、`vulnerability.json`、`manifest.json`
- CI：每次爬虫/处理成功后，自动在 Git 发布 manifest，与 checksum 同步
- 可通过 Webhook 通知 SEIA-Mod 仓库（或手动执行 `make asset-bundle`）

## 5. 接口留白与后续扩展

- **协议扩展**：manifest 可新增 `api_endpoint` 字段，指向 Asset Hub 暴露的 REST/gRPC 服务；本仓随后实现 `RemoteBundleProvider`。
- **监控**：在 SEIA-Mod 中记录已导入 bundle 的 `bundle_id + version`，并将其写入 ArangoDB `Asset` 文档的 `bundle_version` 字段，便于回滚。
- **多租户**：若需多个外部团队提供资产包，可在 manifest 层增加 `provider` 字段并在 `.env` 中指定默认 provider。

通过以上策略，SEIA-Mod 能保持核心框架简洁，仅需消费标准化资产包；Asset Hub 则可自由迭代爬虫、遥感与清洗逻辑，双方通过 manifest+Provider 接口完成集成。
