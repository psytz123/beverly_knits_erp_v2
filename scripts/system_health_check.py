#!/usr/bin/env python3
"""System health verification for Beverly Knits ERP dashboard services.

Validates that the API server exposes live eFab data and that dashboard
endpoints respond successfully. Exits with non-zero status upon failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict

try:
    import requests
except ImportError as exc:  # pragma: no cover - handled by runtime usage
    print("ERROR: 'requests' library is required. Install with 'pip install requests'.", file=sys.stderr)
    raise SystemExit(1) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ERP API and dashboard health endpoints.")
    parser.add_argument("--host", default="http://localhost", help="Base host for the API server (default: http://localhost)")
    parser.add_argument("--port", type=int, default=5006, help="API server port (default: 5006)")
    parser.add_argument("--retries", type=int, default=3, help="Number of retry attempts before failing")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds to wait between retries")
    return parser.parse_args()


def build_url(host: str, port: int, path: str) -> str:
    host = host.rstrip('/')
    path = path.lstrip('/')
    return f"{host}:{port}/{path}"


def fetch_json(url: str) -> Dict[str, Any]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Non-JSON response from {url}") from exc


def validate_health(health_payload: Dict[str, Any]) -> None:
    if health_payload.get("status") != "healthy":
        raise RuntimeError("Health endpoint did not return status=healthy")
    if health_payload.get("data_source") != "efab_direct":
        raise RuntimeError("Health endpoint indicates non-eFab data source")
    if not health_payload.get("efab_connected"):
        raise RuntimeError("eFab connection reported as inactive")


def validate_yarn_payload(payload: Any) -> None:
    if not isinstance(payload, list):
        raise RuntimeError("Yarn intelligence response must be a list")
    if not payload:
        raise RuntimeError("Yarn intelligence response returned no records")
    sample = payload[0]
    required_fields = {"yarn_id", "description", "theoretical_balance", "planning_balance"}
    missing = required_fields - set(sample.keys())
    if missing:
        raise RuntimeError(f"Yarn intelligence payload missing fields: {sorted(missing)}")


def main() -> int:
    args = parse_args()
    health_url = build_url(args.host, args.port, "/api/health")
    yarn_url = build_url(args.host, args.port, "/api/yarn-intelligence")

    for attempt in range(1, args.retries + 1):
        try:
            print(f"Attempt {attempt}/{args.retries}: Checking {health_url}")
            health_payload = fetch_json(health_url)
            validate_health(health_payload)
            print("  ✓ Health endpoint OK (efab_direct confirmed)")

            print(f"Attempt {attempt}/{args.retries}: Checking {yarn_url}")
            yarn_payload = fetch_json(yarn_url)
            validate_yarn_payload(yarn_payload)
            print("  ✓ Yarn intelligence endpoint returning data")

            print("System health verified successfully.")
            return 0
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  ⚠️  Validation failed: {exc}")
            if attempt < args.retries:
                time.sleep(args.interval)

    print("ERROR: System health validation failed after maximum retries.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
