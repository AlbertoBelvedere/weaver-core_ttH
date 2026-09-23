#!/bin/bash

#SBATCH --job-name=weaver_stxs
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00

set -e

usage() {
    echo "Usage: $0 [VAR=value ...] <executable> [arguments ...]"
    echo "Example: $0 NUM_EPOCHS=10 ./execution_scripts/launcher_stxs.sh --demo"
    echo "Example: $0 MODE=test ./execution_scripts/launcher_stxs.sh"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p slurm_logs
    exec sbatch "$0" "$@"
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

JOB_SCRATCH=/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$JOB_SCRATCH"
export TMPDIR="$JOB_SCRATCH"

cleanup_scratch() {
    if [[ -n "${JOB_SCRATCH:-}" && "$JOB_SCRATCH" == /scratch/* ]]; then
        rm -rf "$JOB_SCRATCH"
    fi
}
trap cleanup_scratch EXIT

env_args=()
while [[ "$#" -gt 0 && "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    export "$1"
    env_args+=("$1")
    shift
done

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Workdir: $(pwd)"
echo "Scratch: ${JOB_SCRATCH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
if [ "${#env_args[@]}" -gt 0 ]; then
    echo "Environment overrides: ${env_args[*]}"
fi
echo "Command: $*"

"$@"
