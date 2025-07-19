#!/usr/bin/env bash

# Usage:
# bash scripts/install_uv.sh [...]

# Usage (as a source):
# __SOURCE_ONLY__=1 source scripts/install_uv.sh [...]
# run_prepare
# run_install
# download_weights

SCRIPT_PATH="${BASH_SOURCE[0]:-${0}}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${SCRIPT_PATH}")" &> /dev/null && pwd)

PACKAGE_MANAGER="uv"
__SOURCE_ONLY__="${__SOURCE_ONLY__:-0}"

if [[ "${__SOURCE_ONLY__}" -ne 0 ]]; then
    PACKAGE_MANAGER="${PACKAGE_MANAGER}" \
    __SOURCE_ONLY__="${__SOURCE_ONLY__}" \
    source "${SCRIPT_DIR}/install.sh" "$@"
else
    env \
        PACKAGE_MANAGER="${PACKAGE_MANAGER}" \
        __SOURCE_ONLY__="${__SOURCE_ONLY__}" \
        "${SCRIPT_DIR}/install.sh" "$@"
fi
