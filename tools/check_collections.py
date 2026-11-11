#!/usr/bin/env python3
"""Read-only checks of exact collection artifacts and all-pair evidence."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collection_registry import Collection, discover

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=ROOT)
parser.add_argument("--collection", action="append")
args = parser.parse_args()
ids = args.collection or discover(args.root)
results, errors = [], []
for identity in ids:
    try:
        results.append(Collection(args.root, identity).verify())
    except Exception as exc:
        errors.append({"collection_id": identity, "error": str(exc)})
print(json.dumps({"collections": results, "errors": errors}, indent=2))
raise SystemExit(bool(errors))
