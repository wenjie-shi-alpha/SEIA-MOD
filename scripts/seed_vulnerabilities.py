#!/usr/bin/env python
"""Seed vulnerability curve definitions into ArangoDB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from arango import ArangoClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("resources/vulnerability/jjj_power_vulnerabilities.json"),
        help="Path to the vulnerability JSON file.",
    )
    parser.add_argument("--arangodb-url", default="http://localhost:8529")
    parser.add_argument("--database", default="seia_mod")
    parser.add_argument("--username", default="root")
    parser.add_argument("--password", default="")
    parser.add_argument(
        "--collection",
        default="Vulnerability",
        help="Target ArangoDB collection for vulnerability curves.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Enable TLS verification when connecting to ArangoDB.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Vulnerability file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Expected top-level list of vulnerability documents.")
    return payload


def ensure_collection(db, name: str):
    if not db.has_collection(name):
        return db.create_collection(name)
    return db.collection(name)


def main() -> None:
    args = parse_args()
    docs = load_json(args.file)
    client = ArangoClient(hosts=args.arangodb_url)
    db = client.db(
        args.database,
        username=args.username,
        password=args.password,
        verify=args.verify,
    )
    collection = ensure_collection(db, args.collection)
    collection.import_bulk(docs, on_duplicate="update")
    print(f"Upserted {len(docs)} vulnerability curves into {args.collection}.")


if __name__ == "__main__":
    main()
