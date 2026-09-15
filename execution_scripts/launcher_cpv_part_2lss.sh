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

# Input ntuples produced by produce_ntuples_train_tth_cpv.py.
DATADIR=${DATADIR:-/work/abelvede/multilepton-analysis/neuralnetwork/snapshots/cpv_part_2lss/2026-08-21}

# Runtime settings.
# Default baseline: eventmlp_fullsplit, batch 512, LR 1e-4. This is favored
# over largerfc because it has nearly the same held-out AUC with less overtraining.
batch_size=${BATCH:-512}
gpu=${GPU:-0}
extra_args=("${@:1}")
start_lr=${START_LR:-1e-4}
num_epochs=${NUM_EPOCHS:-50}
weight_mode=${WEIGHT_MODE:-fullweight_bdtvars}
data_config=${DATA_CONFIG:-data/cpv_part_2lss_${weight_mode}.yaml}
loss_tag=${LOSS_TAG:-classbalancedloss}
model_variant=${MODEL_VARIANT:-eventmlp}

case "${model_variant}" in
    largerfc)
        network_config=${NETWORK_CONFIG:-data/cpv_part_2lss_model.py}
        model_tag=${MODEL_TAG:-largerfc}
        ;;
    eventmlp)
        network_config=${NETWORK_CONFIG:-data/cpv_part_2lss_eventmlp_model.py}
        model_tag=${MODEL_TAG:-eventmlp_fullsplit_lr1em4}
        ;;
    eventmlpwide)
        network_config=${NETWORK_CONFIG:-data/cpv_part_2lss_eventmlp_wide_model.py}
        model_tag=${MODEL_TAG:-eventmlpwide}
        ;;
    objectonly|noevent)
        network_config=${NETWORK_CONFIG:-data/cpv_part_2lss_objectonly_model.py}
        model_tag=${MODEL_TAG:-objectonly}
        ;;
    custom)
        network_config=${NETWORK_CONFIG:?Set NETWORK_CONFIG when MODEL_VARIANT=custom}
        model_tag=${MODEL_TAG:-custom}
        ;;
    *)
        echo "Unknown MODEL_VARIANT: ${model_variant}" >&2
        echo "Use MODEL_VARIANT=largerfc, MODEL_VARIANT=eventmlp, MODEL_VARIANT=eventmlpwide, MODEL_VARIANT=objectonly, or MODEL_VARIANT=custom with NETWORK_CONFIG=..." >&2
        exit 1
        ;;
esac

if [[ ! -f "${data_config}" ]]; then
    echo "Data config not found: ${data_config}" >&2
    echo "Use WEIGHT_MODE=fullweight_bdtvars, WEIGHT_MODE=cpmodelweight_bdtvars, or set DATA_CONFIG explicitly." >&2
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
training_name=cpv_part_2lss_${weight_mode}_${loss_tag}_${model_tag}_fetch1_batch${batch_size}_lr${lr_tag}

mkdir -p result/cpv_part_2lss logs_full_model

python weaver/train.py --data-train \
    "cp_even:${DATADIR}/*/TTH_ctcvcp*_cp_even.root" \
    "cp_odd:${DATADIR}/*/TTH_ctcvcp*_cp_odd.root" \
    --data-val \
    "cp_even:${DATADIR}/*/TTH_ctcvcp*_cp_even.root" \
    "cp_odd:${DATADIR}/*/TTH_ctcvcp*_cp_odd.root" \
    --fetch-step 1 --batch-size "${batch_size}" --start-lr "${start_lr}" \
    --data-config "${data_config}" --network-config "${network_config}" \
    --num-epochs "${num_epochs}" \
    --model-prefix "result/cpv_part_2lss/${training_name}/fullmodel" \
    --gpus "${gpu}" \
    --optimizer ranger \
    --log-file "logs_full_model/log_fullmodel_${training_name}.log" \
    --tensorboard "${training_name}" \
    --extra-selection-train 'event_mod5 < 3' \
    --extra-selection-val 'event_mod5 == 3' \
    "${extra_args[@]}"
