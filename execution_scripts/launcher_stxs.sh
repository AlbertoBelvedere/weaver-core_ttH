#!/bin/bash
set -euo pipefail

source /work/abelvede/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-weaver}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

mode=${MODE:-train}
data_glob=${DATA_GLOB:-/work/abelvede/multilepton-analysis/neuralnetwork/snapshots/cpv_part_2lss/2026-08-21/*/TTH_ctcvcp*_cp_*.root}
data_config=${DATA_CONFIG:-data/stxs.yaml}
network_config=${NETWORK_CONFIG:-data/stxs_model.py}
run_name=${RUN_NAME:-baseline}
model_prefix=${MODEL_PREFIX:-result/stxs/${run_name}/fullmodel}
batch_size=${BATCH:-512}
gpu=${GPU-0}

[[ -f ${data_config} ]] || { echo "Data config not found: ${data_config}" >&2; exit 1; }
[[ -f ${network_config} ]] || { echo "Model config not found: ${network_config}" >&2; exit 1; }
compgen -G "${data_glob}" > /dev/null || { echo "No input files match DATA_GLOB=${data_glob}" >&2; exit 1; }

mkdir -p "$(dirname "${model_prefix}")" logs_stxs

case "${mode}" in
    train)
        python weaver/train.py --run-mode train,val \
            --data-train "sample:${data_glob}" \
            --data-val "sample:${data_glob}" \
            --data-config "${data_config}" --network-config "${network_config}" \
            --model-prefix "${model_prefix}" \
            --fetch-step 1 --batch-size "${batch_size}" \
            --start-lr "${START_LR:-1e-4}" --num-epochs "${NUM_EPOCHS:-50}" \
            --gpus "${gpu}" --optimizer ranger \
            --log-file "logs_stxs/${run_name}_train.log" \
            --extra-selection-train 'event_mod5 < 3' \
            --extra-selection-val 'event_mod5 == 3' \
            "$@"
        ;;
    test)
        python weaver/train.py --run-mode test \
            --data-test "sample:${data_glob}" \
            --data-config "${data_config}" --network-config "${network_config}" \
            --model-prefix "${model_prefix}" \
            --predict-output "${PREDICT_OUTPUT:-result/stxs/${run_name}/predictions.root}" \
            --batch-size "${batch_size}" --gpus "${gpu}" \
            --log-file "logs_stxs/${run_name}_test.log" \
            "$@"
        ;;
    *)
        echo "MODE must be train or test" >&2
        exit 1
        ;;
esac
