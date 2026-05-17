#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
ENV_FILE="${REPO_ROOT}/.env"

HOST="127.0.0.1"
PORT="8000"
USE_NGROK="0"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/start_web.sh [options]

Options:
  --host HOST      Host to bind locally. Default: 127.0.0.1
  --port PORT      Port to bind locally. Default: 8000
  --ngrok          Expose the local UI through ngrok.
  -h, --help       Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:?--host requires a value}"
      shift 2
      ;;
    --port)
      PORT="${2:?--port requires a value}"
      shift 2
      ;;
    --ngrok)
      USE_NGROK="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtual environment Python not found: ${PYTHON_BIN}" >&2
  echo "Create .venv before starting the web UI." >&2
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
fi

CMD=("${PYTHON_BIN}" "${REPO_ROOT}/web_app.py" --host "${HOST}" --port "${PORT}")

if [[ "${USE_NGROK}" == "1" ]]; then
  if ! command -v ngrok >/dev/null 2>&1; then
    echo "ngrok was requested but was not found on PATH." >&2
    exit 1
  fi
  CMD+=(--ngrok --ngrok-bin "$(command -v ngrok)")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

cd "${REPO_ROOT}"
exec "${CMD[@]}"
