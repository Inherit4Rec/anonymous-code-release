#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_PYTHON="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"

"${BASE_PYTHON}" -c 'import sys; raise SystemExit(sys.version_info < (3, 9))' || {
    echo "Python 3.9 or newer is required." >&2
    exit 1
}
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${BASE_PYTHON}" -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT}/requirements.txt"
echo "Dependencies installed in ${VENV_DIR}"
