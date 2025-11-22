"""Graph loader for ArangoDB exposure schema."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import geopandas as gpd
from arango import ArangoClient
from geopandas import GeoDataFrame
import h3

logger = logging.getLogger(__name__)


@dataclass
class GraphLoaderConfig:
    """Configuration for loading exposure assets into ArangoDB."""

    geo_parquet_path: Path
    arangodb_url: str
    database: str
    username: str
    password: str
    asset_collection: str = "Assets"
    supply_chain_collection: str = "SupplyChain"
    vulnerability_collection: str = "Vulnerability"
    vulnerability_edge_collection: str = "HasVulnerability"
    supplier_field: str = "supplier_ids"
    vulnerability_field: str = "vulnerability_type"
    h3_resolution: int = 9
    asset_key_field: str = "_key"
    batch_size: int = 1000
    vulnerability_mapping: Dict[str, str] | None = field(default_factory=dict)


class ExposureGraphLoader:
    """Transforms GeoParquet asset data into ArangoDB collections."""

    def __init__(self, config: GraphLoaderConfig) -> None:
        self.config = config
        self.config.geo_parquet_path = Path(self.config.geo_parquet_path)
        self._db = None

    @property
    def db(self):
        if self._db is None:
            client = ArangoClient(hosts=self.config.arangodb_url)
            self._db = client.db(
                self.config.database,
                username=self.config.username,
                password=self.config.password,
                verify=True,
            )
        return self._db

    def _ensure_collection(self, name: str, edge: bool = False):
        if not self.db.has_collection(name):
            self.db.create_collection(name, edge=edge)
        return self.db.collection(name)

    def ensure_indexes(self) -> None:
        """Create indexes to accelerate graph queries."""
        assets = self._ensure_collection(self.config.asset_collection)
        assets.add_hash_index(fields=["h3_index"], unique=False, sparse=True)
        assets.add_hash_index(fields=["type"], unique=False, sparse=True)
        edges = self._ensure_collection(self.config.supply_chain_collection, edge=True)
        edges.add_hash_index(fields=["_from", "_to"], unique=True, sparse=False)

    def load_assets(self) -> List[dict]:
        """Read GeoParquet, compute H3 indices, and prepare payloads."""
        gdf = self._read_geo_parquet()
        if gdf.empty:
            logger.warning("GeoParquet file %s is empty.", self.config.geo_parquet_path)
            return []
        gdf = self._ensure_crs(gdf)
        gdf["h3_index"] = gdf.geometry.centroid.apply(
            lambda geom: self._geo_to_h3(geom.y, geom.x, self.config.h3_resolution)
        )
        geometries = gdf.geometry.to_wkt()
        frame = gdf.drop(columns="geometry").copy()
        frame["geometry_wkt"] = geometries
        records = frame.to_dict(orient="records")
        for idx, doc in enumerate(records):
            doc.setdefault(self.config.asset_key_field, f"{doc['h3_index']}_{idx}")
            doc["_key"] = str(doc[self.config.asset_key_field])
            doc["h3_index"] = str(doc["h3_index"])
        return records

    def _read_geo_parquet(self) -> GeoDataFrame:
        path = self.config.geo_parquet_path
        if not path.exists():
            raise FileNotFoundError(f"GeoParquet file not found: {path}")
        logger.info("Reading GeoParquet asset file: %s", path)
        return gpd.read_parquet(path)

    @staticmethod
    def _ensure_crs(gdf: GeoDataFrame) -> GeoDataFrame:
        if gdf.crs is None:
            logger.warning("GeoDataFrame CRS missing; assuming EPSG:4326.")
            return gdf.set_crs(epsg=4326)
        if gdf.crs.to_epsg() != 4326:
            return gdf.to_crs(epsg=4326)
        return gdf

    def upsert_assets(self, docs: Sequence[dict]) -> None:
        """Write assets into the ArangoDB asset collection."""
        if not docs:
            logger.warning("No asset documents to write.")
            return
        collection = self._ensure_collection(self.config.asset_collection)
        logger.info("Importing %d assets into %s", len(docs), self.config.asset_collection)
        for chunk in self._chunk(docs):
            collection.import_bulk(chunk, on_duplicate="update")

    def build_supply_chain_edges(self, docs: Sequence[dict]) -> None:
        """Create SupplyChain edges based on logistics metadata."""
        edges = []
        for doc in docs:
            suppliers = self._normalize_list(doc.get(self.config.supplier_field))
            for supplier in suppliers:
                edges.append(
                    {
                        "_from": f"{self.config.asset_collection}/{supplier}",
                        "_to": f"{self.config.asset_collection}/{doc['_key']}",
                        "type": "logistics",
                    }
                )
        if not edges:
            logger.info("No supply chain edges detected in dataset.")
            return
        collection = self._ensure_collection(self.config.supply_chain_collection, edge=True)
        logger.info("Importing %d supply chain edges", len(edges))
        for chunk in self._chunk(edges):
            collection.import_bulk(chunk, on_duplicate="ignore")

    def bind_vulnerabilities(self, docs: Sequence[dict]) -> None:
        """Associate assets to vulnerability curves."""
        mapping = self.config.vulnerability_mapping or {}
        edges = []
        for doc in docs:
            vtype = doc.get(self.config.vulnerability_field)
            if not vtype:
                continue
            vulnerability_key = mapping.get(vtype, vtype)
            edges.append(
                {
                    "_from": f"{self.config.asset_collection}/{doc['_key']}",
                    "_to": f"{self.config.vulnerability_collection}/{vulnerability_key}",
                }
            )
        if not edges:
            logger.info("No vulnerability bindings to create.")
            return
        collection = self._ensure_collection(self.config.vulnerability_edge_collection, edge=True)
        logger.info("Importing %d vulnerability edges", len(edges))
        for chunk in self._chunk(edges):
            collection.import_bulk(chunk, on_duplicate="ignore")

    def impacted_assets_aql(self, h3_ids: Iterable[str]) -> str:
        """Return an AQL snippet for querying impacted assets by H3 ids."""
        placeholders = ", ".join(f'"{h3_id}"' for h3_id in h3_ids)
        return (
            f"FOR asset IN {self.config.asset_collection} "
            f"FILTER asset.h3_index IN [{placeholders}] "
            "RETURN asset"
        )

    def run(self) -> None:
        assets = self.load_assets()
        self.ensure_indexes()
        self.upsert_assets(assets)
        self.build_supply_chain_edges(assets)
        self.bind_vulnerabilities(assets)

    def _chunk(self, docs: Sequence[dict]) -> Iterable[List[dict]]:
        batch = []
        for doc in docs:
            batch.append(doc)
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _normalize_list(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, Iterable):
            return [str(item) for item in value if item]
        return []

    @staticmethod
    def _geo_to_h3(lat: float, lon: float, resolution: int) -> str:
        if hasattr(h3, "geo_to_h3"):
            return h3.geo_to_h3(lat, lon, resolution)
        return h3.latlng_to_cell(lat, lon, resolution)


__all__ = ["ExposureGraphLoader", "GraphLoaderConfig"]
