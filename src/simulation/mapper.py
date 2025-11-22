"""Dynamic spatio-temporal mapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Protocol, Sequence

from arango import ArangoClient


@dataclass
class MapperConfig:
    """Configuration for DynamicMapper and its ArangoDB backend."""

    arangodb_url: str
    database: str
    username: str
    password: str
    neighborhood_hops: int = 2
    asset_collection: str = "Assets"
    supply_chain_collection: str = "SupplyChain"
    vulnerability_collection: str = "Vulnerability"
    vulnerability_edge_collection: str = "HasVulnerability"


class GraphBackendProtocol(Protocol):
    """Backend abstraction for fetching subgraphs."""

    def assets_by_h3(self, h3_ids: Sequence[str]) -> List[Dict[str, Any]]:  # pragma: no cover - protocol
        ...

    def vulnerability_context(self, asset_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:  # pragma: no cover
        ...

    def subgraph_from_assets(self, asset_keys: Sequence[str], hops: int) -> Dict[str, Any]:  # pragma: no cover
        ...


class ArangoGraphBackend:
    """Graph backend implementation backed by ArangoDB collections."""

    def __init__(self, config: MapperConfig) -> None:
        self.config = config
        client = ArangoClient(hosts=config.arangodb_url)
        self.db = client.db(
            config.database,
            username=config.username,
            password=config.password,
            verify=True,
        )

    def assets_by_h3(self, h3_ids: Sequence[str]) -> List[Dict[str, Any]]:
        query = (
            f"FOR asset IN {self.config.asset_collection} "
            "FILTER asset.h3_index IN @h3_ids "
            "RETURN MERGE(asset, {__id: asset._id})"
        )
        return list(self.db.aql.execute(query, bind_vars={"h3_ids": list(h3_ids)}))

    def vulnerability_context(self, asset_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not asset_keys:
            return {}
        from_ids = [f"{self.config.asset_collection}/{key}" for key in asset_keys]
        query = (
            f"FOR edge IN {self.config.vulnerability_edge_collection} "
            "FILTER edge._from IN @from_ids "
            f"FOR vuln IN {self.config.vulnerability_collection} "
            "FILTER vuln._id == edge._to "
            "LET asset_key = PARSE_IDENTIFIER(edge._from).key "
            "RETURN {asset_key: asset_key, vulnerability: vuln}"
        )
        context: Dict[str, Dict[str, Any]] = {}
        for row in self.db.aql.execute(query, bind_vars={"from_ids": from_ids}):
            context[row["asset_key"]] = row["vulnerability"]
        return context

    def subgraph_from_assets(self, asset_keys: Sequence[str], hops: int) -> Dict[str, Any]:
        """Return nodes/edges for up to N-hop neighborhood around the assets."""
        if not asset_keys:
            return {"nodes": [], "edges": []}
        doc_ids = [f"{self.config.asset_collection}/{key}" for key in asset_keys]
        query = (
            f"FOR v, e, p IN 0..@hops ANY @start_ids "
            f"GRAPH '{self._graph_name}' "
            "FILTER IS_SAME_COLLECTION(@asset_collection, v) "
            "RETURN {{vertex: v, edge: e}}"
        )
        # The above query expects a named graph. To avoid forcing schema, fall back to manual expansion
        # when the named graph is absent.
        try:
            cursor = self.db.aql.execute(
                query,
                bind_vars={
                    "hops": hops,
                    "start_ids": doc_ids,
                    "asset_collection": self.config.asset_collection,
                },
            )
            nodes: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []
            for row in cursor:
                vertex = row.get("vertex")
                if vertex:
                    nodes[vertex["_key"]] = vertex
                edge = row.get("edge")
                if edge:
                    edges.append(edge)
            return {"nodes": list(nodes.values()), "edges": edges}
        except Exception:
            return self._manual_subgraph(doc_ids, hops)

    @property
    def _graph_name(self) -> str:
        """Derive a deterministic name for the Arango graph."""
        return f"{self.config.database}_exposure"

    def _manual_subgraph(self, doc_ids: Sequence[str], hops: int) -> Dict[str, Any]:
        """Fallback approach that manually expands SupplyChain edges."""
        nodes: Dict[str, Dict[str, Any]] = {}
        visited: set[str] = set()
        assets_cursor = self.db.aql.execute(
            f"FOR asset IN {self.config.asset_collection} "
            "FILTER asset._id IN @ids "
            "RETURN asset",
            bind_vars={"ids": list(doc_ids)},
        )
        for doc in assets_cursor:
            nodes[doc["_key"]] = doc
            visited.add(doc["_id"])
        frontier = set(doc_ids)
        edges: List[Dict[str, Any]] = []
        assets_collection = self.db.collection(self.config.asset_collection)
        for _ in range(hops):
            if not frontier:
                break
            cursor = self.db.aql.execute(
                f"FOR edge IN {self.config.supply_chain_collection} "
                "FILTER edge._from IN @frontier OR edge._to IN @frontier "
                "RETURN edge",
                bind_vars={"frontier": list(frontier)},
            )
            new_frontier: set[str] = set()
            for edge in cursor:
                edges.append(edge)
                new_frontier.update([edge["_from"], edge["_to"]])
                for endpoint in (edge["_from"], edge["_to"]):
                    doc = assets_collection.get(endpoint)
                    if doc:
                        nodes.setdefault(doc["_key"], doc)
            frontier = new_frontier - visited
            visited.update(new_frontier)
        return {"nodes": list(nodes.values()), "edges": edges}


class DynamicMapper:
    """Resolves event coverage into a simulation-ready subgraph."""

    def __init__(self, config: MapperConfig, backend: GraphBackendProtocol | None = None) -> None:
        self.config = config
        self.backend = backend or ArangoGraphBackend(config)

    def fetch_impacted_assets(self, h3_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = [h for h in h3_ids if h]
        if not ids:
            return []
        return self.backend.assets_by_h3(ids)

    def attach_vulnerability_context(self, assets: List[Dict[str, Any]]) -> None:
        keys = [asset["_key"] for asset in assets]
        context = self.backend.vulnerability_context(keys)
        for asset in assets:
            asset["vulnerability"] = context.get(asset["_key"], asset.get("vulnerability", {}))

    def expand_subgraph(self, assets: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = [asset["_key"] for asset in assets]
        subgraph = self.backend.subgraph_from_assets(keys, self.config.neighborhood_hops)
        # Ensure impacted assets are present and override duplicates with enriched versions.
        keyed_nodes = {node.get("_key"): node for node in subgraph.get("nodes", []) if node.get("_key")}
        for asset in assets:
            keyed_nodes[asset["_key"]] = asset
        subgraph["nodes"] = list(keyed_nodes.values())
        return subgraph

    def run(self, event_payload: Dict[str, Any]) -> Dict[str, Any]:
        h3_ids = event_payload.get("h3_coverage", [])
        assets = self.fetch_impacted_assets(h3_ids)
        if not assets:
            return {"nodes": [], "edges": []}
        self.attach_vulnerability_context(assets)
        return self.expand_subgraph(assets)


__all__ = ["DynamicMapper", "MapperConfig", "GraphBackendProtocol", "ArangoGraphBackend"]
