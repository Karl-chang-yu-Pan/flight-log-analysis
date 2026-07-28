#!/venv/bin/python
"""Client executable for commit-pinned Git reads inside the shell sandbox."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path


BROKER_TIMEOUT_S = 1200


def main() -> int:
    broker_value = os.environ.get("FLIGHT_LOG_SOURCE_BROKER")
    if not broker_value:
        print(
            "PX4 source access is unavailable in this Python stage.",
            file=sys.stderr,
        )
        return 126

    broker = Path(broker_value)
    request_id = uuid.uuid4().hex
    request_path = broker / "requests" / f"request-{request_id}.json"
    response_path = broker / "responses" / f"response-{request_id}.json"
    temporary_request = broker / "requests" / f".request-{request_id}.tmp"
    try:
        temporary_request.write_text(
            json.dumps({"argv": sys.argv[1:]}),
            encoding="utf-8",
        )
        temporary_request.replace(request_path)

        deadline = time.monotonic() + BROKER_TIMEOUT_S
        while not response_path.is_file():
            if time.monotonic() >= deadline:
                print("Snapshot Git broker timed out.", file=sys.stderr)
                return 124
            time.sleep(0.01)

        response = json.loads(response_path.read_text(encoding="utf-8"))
        sys.stdout.write(str(response.get("stdout") or ""))
        sys.stderr.write(str(response.get("stderr") or ""))
        return int(response.get("returncode", 126))
    finally:
        temporary_request.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
