#!/usr/bin/env bash
set -euo pipefail

echo "Running unit tests (production/tests/unit)…"
pytest -q production/tests/unit || pytest -q -k benchmark_loader production/tests/unit


