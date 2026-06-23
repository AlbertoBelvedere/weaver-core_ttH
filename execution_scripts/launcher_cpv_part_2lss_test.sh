#!/bin/bash
#Four combination using 
#WEIGHT_MODE=fullweight_bdtvars or WEIGHT_MODE=cpmodelweight_bdtvars,  
#AND
#MODEL_VARIANT=largerfc or MODEL_VARIANT=eventmlp
#each of the four should be trained both for 
#EVAL_SPLIT=test and EVAL_SPLIT=train.

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
DATADIR=${DATADIR:-/work/abelvede/multilepton-analysis/neuralnetwork/snapshots/cpv_part_2lss/2026-05-28}

# Runtime settings. Keep these consistent with the training launcher unless
# MODEL_PREFIX is set explicitly.
batch_size=${BATCH:-512}
gpu=${GPU:-0}
extra_args=("${@:1}")
start_lr=${START_LR:-1e-2}
weight_mode=${WEIGHT_MODE:-fullweight_bdtvars}
data_config=${DATA_CONFIG:-data/cpv_part_2lss_${weight_mode}.yaml}
loss_tag=${LOSS_TAG:-classbalancedloss}
model_variant=${MODEL_VARIANT:-largerfc}
eval_split=${EVAL_SPLIT:-test}

case "${model_variant}" in
    largerfc)
        network_config=${NETWORK_CONFIG:-data/cpv_part_2lss_model.py}
        model_tag=${MODEL_TAG:-largerfc}
        ;;
    eventmlp)
        network_config=${NETWORK_CONFIG:-data/cpv_part_2lss_eventmlp_model.py}
        model_tag=${MODEL_TAG:-eventmlp}
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

case "${eval_split}" in
    test)
        test_time_selection='event_mod5 == 4'
        predict_stem='predict_output_test'
        ;;
    train)
        test_time_selection='event_mod5 < 3'
        predict_stem='predict_output_train'
        ;;
    val|validation)
        test_time_selection='event_mod5 == 3'
        predict_stem='predict_output_val'
        eval_split='val'
        ;;
    *)
        echo "Unknown EVAL_SPLIT: ${eval_split}" >&2
        echo "Use EVAL_SPLIT=test, EVAL_SPLIT=train, or EVAL_SPLIT=val." >&2
        exit 1
        ;;
esac

lr_tag=$(printf "%.12g" "${start_lr}")
lr_tag=${lr_tag//./p}
lr_tag=${lr_tag//-/m}
lr_tag=${lr_tag//+/p}
training_name=cpv_part_2lss_${weight_mode}_${loss_tag}_${model_tag}_fetch1_batch${batch_size}_lr${lr_tag}

model_prefix=${MODEL_PREFIX:-result/cpv_part_2lss/${training_name}/fullmodel}
predict_output=${PREDICT_OUTPUT:-result/cpv_part_2lss/${training_name}/${predict_stem}.root}

mkdir -p result/cpv_part_2lss/"${training_name}" logs_full_model

if [[ "${eval_split}" != "test" ]]; then
    split_data_config="result/cpv_part_2lss/${training_name}/$(basename "${data_config}" .yaml)_${eval_split}_infer.yaml"
    python - "${data_config}" "${split_data_config}" "${test_time_selection}" <<'PY'
import hashlib
import re
import sys
from pathlib import Path

src, dst, selection = sys.argv[1:]
src_path = Path(src)
source_config = src_path
if ".auto.yaml" not in src_path.name:
    digest = hashlib.md5(src_path.read_bytes()).hexdigest()
    auto_config = src_path.with_name(src_path.name.replace(".yaml", f".{digest}.auto.yaml"))
    if auto_config.exists():
        source_config = auto_config

text = source_config.read_text()
replacement = f"test_time_selection:\n   {selection}\n\n"
text, count = re.subn(
    r"(?ms)^test_time_selection:\n(?:^[ \t]+.*\n?)+",
    replacement,
    text,
    count=1,
)
if count != 1:
    text, count = re.subn(
        r"(?m)^test_time_selection:.*$",
        replacement.rstrip(),
        text,
        count=1,
    )
if count != 1:
    raise RuntimeError(f"Could not replace test_time_selection in {source_config}")
Path(dst).write_text(text)
PY
    data_config="${split_data_config}"
fi

python weaver/train.py --run-mode test \
    --data-test \
    "cp_even:${DATADIR}/*/TTH_ctcvcp_sm_cp_even.root" \
    "cp_odd:${DATADIR}/*/TTH_ctcvcp_sm_cp_odd.root" \
    --batch-size "${batch_size}" \
    --data-config "${data_config}" --network-config "${network_config}" \
    --model-prefix "${model_prefix}" \
    --predict-output "${predict_output}" \
    --gpus "${gpu}" \
    --log-file "logs_full_model/log_test_${training_name}_${eval_split}.log" \
    "${extra_args[@]}"
