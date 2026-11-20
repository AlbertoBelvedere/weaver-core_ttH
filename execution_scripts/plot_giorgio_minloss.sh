#!/bin/bash
# Regenerate train/test evaluation plots for all Giorgio ntuple runs.
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Slurm executes a copied script from its spool directory; --chdir sets PWD.
    repo_dir=${PWD}
else
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${repo_dir}"
python_bin=${PYTHON_BIN:-/work/abelvede/miniconda3/envs/weaver/bin/python}

runs=(
    cpv_part_2lss_SR_2lss_classbalancedloss_eventmlp_ttH_fetch1_batch512_lr0p0001
    cpv_part_2lss_SR_2lss_classbalancedloss_eventtoken_ttH_fetch1_batch512_lr0p0001
    cpv_part_2lss_SR_2lss_extended_classbalancedloss_eventmlp_ttH_fetch1_batch512_lr0p0001
    cpv_part_2lss_SR_2lss_extended_classbalancedloss_eventtoken_ttH_fetch1_batch512_lr0p0001
    cpv_part_2lss_SR_2lss_classbalancedloss_eventtoken3_fetch1_batch512_lr0p0001
    cpv_part_3l_SR_3l_classbalancedloss_eventmlp_fetch1_batch512_lr0p0001
    cpv_part_3l_SR_3l_classbalancedloss_eventtoken_fetch1_batch512_lr0p0001
    cpv_part_3l_SR_3l_classbalancedloss_eventtoken3_fetch1_batch512_lr0p0001
)

for name in "${runs[@]}"; do
    channel=${name#cpv_part_}
    channel=${channel%%_*}
    run_dir="result/cpv_part_${channel}/giorgio_ntuples/${name}"
    for split in train test; do
        for class in cp_even cp_odd; do
            file="${run_dir}/predict_output_${split}_${class}.root"
            if [[ ! -s "${file}" ]]; then
                echo "Missing prediction file: ${file}" >&2
                exit 1
            fi
        done
    done
    "${python_bin}" evaluation/evaluate_cpv_part_2lss.py \
        --input-dir "${run_dir}" \
        --train-input-dir "${run_dir}" \
        --class-weight-branches weight_SM weight_CP_odd \
        --log-file "logs_full_model/log_fullmodel_${name}.log" \
        --output "${run_dir}/evaluation"
done
