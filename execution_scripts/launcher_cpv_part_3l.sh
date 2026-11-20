#!/bin/bash

# Three-lepton entry point; the shared launcher selects the 3l data and output paths.
export DATA_VARIANT=SR_3l
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/launcher_cpv_part_2lss.sh" "$@"
