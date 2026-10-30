#!/bin/sh
# Resource guard for isolated Drake labeling. Does not alter pose/contact math.
set -eu
acp_project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
exec timeout --signal=TERM --kill-after=5s "${ACP_LABEL_TIMEOUT_SECONDS:-120}" \
    "${ACP_LABEL_REAL_PYTHON:-$acp_project_dir/.venv/bin/python}" "$@"
