"""Standalone developer diagnostics runner.
Usage:
    python dev_backend_report.py sales.csv
Outputs a concise JSON diagnostics file without exposing backend logic in the main UI.
"""
import json
import sys
from customer_analyzer_backend import run_pipeline, serializable_dev

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python dev_backend_report.py <sales.csv>")
    with open(sys.argv[1], "rb") as f:
        result = run_pipeline(f.read())
    out = serializable_dev(result["dev"])
    with open("dev_backend_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("Created dev_backend_results.json")
