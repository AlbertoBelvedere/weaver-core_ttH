#!/bin/bash

set -e

# Source the conda environment
source /work/abelvede/miniconda3/etc/profile.d/conda.sh

# Activate the Weaver conda environment
conda activate "${CONDA_ENV:-weaver}"

# Check the GPU status
nvidia-smi

# Run from the top of the new weaver-core checkout, even if the script is
# launched from another directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEAVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WEAVER_DIR}"

# Runtime settings.
# Default baseline: eventmlp_fullsplit, batch 512, LR 1e-4. This is favored
# over largerfc because it has nearly the same held-out AUC with less overtraining.
batch_size=${BATCH:-512}
gpu=${GPU:-0}
extra_args=("${@:1}")
start_lr=${START_LR:-1e-4}
num_epochs=${NUM_EPOCHS:-50}
# WEIGHT_MODE remains an alias for older commands.
data_variant=${DATA_VARIANT:-${WEIGHT_MODE:-fullweight}}
if [[ "${data_variant}" == SR_2lss || "${data_variant}" == SR_2lss_extended || "${data_variant}" == SR_2lss_1tau || "${data_variant}" == SR_3l ]]; then
    if [[ "${data_variant}" == SR_3l ]]; then
        DATADIR=${DATADIR:-${WEAVER_DIR}/Giorgio_ntuples/SR_3l}
    elif [[ "${data_variant}" == SR_2lss_1tau ]]; then
        DATADIR=${DATADIR:-${WEAVER_DIR}/Giorgio_ntuples/SR_2lss_1tau}
    else
        DATADIR=${DATADIR:-${WEAVER_DIR}/Giorgio_ntuples/SR_2lss}
    fi
    results_group=giorgio_ntuples
else
    DATADIR=${DATADIR:-/work/abelvede/multilepton-analysis/neuralnetwork/snapshots/cpv_part_2lss/2026-08-21}
    results_group=first_setup
fi
case "${data_variant}" in
    SR_2lss) default_data_config=data/configs_2l/cpv_part_2lss_SR.yaml ;;
    SR_2lss_extended) default_data_config=data/configs_2l/cpv_part_2lss_SR_extended.yaml ;;
    SR_2lss_1tau) default_data_config=data/configs_2l_1tau/cpv_part_2lss_1tau_SR.yaml ;;
    SR_3l) default_data_config=data/configs_3l/cpv_part_3l_SR.yaml ;;
    *) default_data_config=data/configs_2l/cpv_part_2lss_${data_variant}.yaml ;;
esac
data_config=${DATA_CONFIG:-${default_data_config}}
loss_tag=${LOSS_TAG:-classbalancedloss}
model_variant=${MODEL_VARIANT:-eventmlp}

case "${model_variant}" in
    largerfc)
        network_config=${NETWORK_CONFIG:-data/models/cpv_part_2lss_model.py}
        model_tag=${MODEL_TAG:-largerfc}
        ;;
    eventmlp)
        network_config=${NETWORK_CONFIG:-data/models/cpv_part_2lss_eventmlp_model.py}
        model_tag=${MODEL_TAG:-eventmlp_fullsplit_lr1em4}
        ;;
    eventtoken)
        network_config=${NETWORK_CONFIG:-data/models/cpv_part_2lss_eventtoken_model.py}
        model_tag=${MODEL_TAG:-eventtoken}
        ;;
    eventtoken3)
        network_config=${NETWORK_CONFIG:-data/models/cpv_part_2lss_eventtoken3_model.py}
        model_tag=${MODEL_TAG:-eventtoken3}
        ;;
    objectonly|noevent)
        network_config=${NETWORK_CONFIG:-data/models/cpv_part_2lss_objectonly_model.py}
        model_tag=${MODEL_TAG:-objectonly}
        ;;
    custom)
        network_config=${NETWORK_CONFIG:?Set NETWORK_CONFIG when MODEL_VARIANT=custom}
        model_tag=${MODEL_TAG:-custom}
        ;;
    *)
        echo "Unknown MODEL_VARIANT: ${model_variant}" >&2
        echo "Use MODEL_VARIANT=largerfc, MODEL_VARIANT=eventmlp, MODEL_VARIANT=eventtoken, MODEL_VARIANT=eventtoken3, MODEL_VARIANT=objectonly, or MODEL_VARIANT=custom with NETWORK_CONFIG=..." >&2
        exit 1
        ;;
esac

if [[ ! -f "${data_config}" ]]; then
    echo "Data config not found: ${data_config}" >&2
    echo "Use DATA_VARIANT=fullweight, DATA_VARIANT=fullweight_2024, DATA_VARIANT=SR_2lss, DATA_VARIANT=SR_2lss_extended, DATA_VARIANT=SR_2lss_1tau, DATA_VARIANT=SR_3l, or set DATA_CONFIG explicitly." >&2
    exit 1
fi

if [[ ! -f "${network_config}" ]]; then
    echo "Network config not found: ${network_config}" >&2
    exit 1
fi

lr_tag=$(printf "%.12g" "${start_lr}")
lr_tag=${lr_tag//./p}
lr_tag=${lr_tag//-/m}
lr_tag=${lr_tag//+/p}
if [[ "${data_variant}" == SR_3l ]]; then
    project_name=cpv_part_3l
else
    project_name=cpv_part_2lss
fi
training_name=${project_name}_${data_variant}_${loss_tag}_${model_tag}_fetch1_batch${batch_size}_lr${lr_tag}
result_dir=result/${project_name}/${results_group}/${training_name}

mkdir -p "${result_dir}" logs_full_model

if [[ "${data_variant}" == SR_2lss || "${data_variant}" == SR_2lss_extended || "${data_variant}" == SR_2lss_1tau || "${data_variant}" == SR_3l ]]; then
    source_dir=$(realpath -e "${DATADIR}")
    shopt -s nullglob
    source_files=("${source_dir}"/TTH_ctcvcp*__*.root)
    shopt -u nullglob
    if (( ${#source_files[@]} != 4 )); then
        echo "Expected four ${data_variant} ROOT files in ${source_dir}; found ${#source_files[@]}" >&2
        exit 1
    fi
    class_links=$(mktemp -d -p "${TMPDIR:-/tmp}" "cpv_${data_variant}_XXXXXX")
    trap 'rm -rf -- "${class_links}"' EXIT
    # The source has no class label; the two paths identify the two weighted views.
    mkdir -p "${class_links}/cp_even" "${class_links}/cp_odd"
    for source_file in "${source_files[@]}"; do
        for class_name in cp_even cp_odd; do
            ln -s "${source_file}" "${class_links}/${class_name}/$(basename "${source_file}")"
        done
    done
    data_files=("cp_even:${class_links}/cp_even/*.root" "cp_odd:${class_links}/cp_odd/*.root")
    # This local ROOT source stalls with uproot's default fsspec reader here.
    train_command=(python -c 'import runpy, uproot; uproot.reading.open.defaults["handler"] = uproot.source.file.MultithreadedFileSource; runpy.run_path("weaver/train.py", run_name="__main__")')
else
    data_files=(
        "cp_even:${DATADIR}/*/TTH_ctcvcp*_cp_even.root"
        "cp_odd:${DATADIR}/*/TTH_ctcvcp*_cp_odd.root"
    )
    train_command=(python weaver/train.py)
fi

"${train_command[@]}" --data-train \
    "${data_files[@]}" \
    --data-val \
    "${data_files[@]}" \
    --fetch-step 1 --batch-size "${batch_size}" --start-lr "${start_lr}" \
    --data-config "${data_config}" --network-config "${network_config}" \
    --num-epochs "${num_epochs}" \
    --model-prefix "${result_dir}/fullmodel" \
    --gpus "${gpu}" \
    --optimizer ranger \
    --log-file "logs_full_model/log_fullmodel_${training_name}.log" \
    --tensorboard "${training_name}" \
    --extra-selection-train 'event_mod5 < 3' \
    --extra-selection-val 'event_mod5 == 3' \
    "${extra_args[@]}"
