#!/bin/bash

# Three-lepton inference entry point; use the matching training run name and path.
export DATA_VARIANT=SR_3l
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/launcher_cpv_part_2lss_test.sh" "$@"
