#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_DIR="${1:-${REPO_ROOT}/data/}"
REMOTE_HOST="${REMOTE_HOST:-MacbookAir}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/Users/jiewenlai/projects/assignment2-systems/data}"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_DATA_DIR}'"
rsync -av --progress "${SOURCE_DIR%/}/" "${REMOTE_HOST}:${REMOTE_DATA_DIR%/}/"
