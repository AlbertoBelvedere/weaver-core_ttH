#!/bin/bash
set -euo pipefail

source /work/inadalli/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-weaver}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

mode=${MODE:-train}
# REGRESSION=1 switches to the pT(H) regressor: regression config and model as defaults,
# and Weaver's --regression-mode (regression training loop, best epoch = lowest validation loss)
regression=${REGRESSION:-0}
if [[ "${regression}" == "1" ]]; then
    default_data_config=data/stxs_regression.yaml
    default_network_config=data/stxs_model_regression.py
    regression_args=(--regression-mode)
else
    default_data_config=data/stxs.yaml
    default_network_config=data/stxs_model.py
    regression_args=()
fi
data_glob=${DATA_GLOB:-/work/abelvede/multilepton-analysis/neuralnetwork/snapshots/STXS/2lss_signal/2026-09-23/*/TTHtoNon2B.root}
data_config=${DATA_CONFIG:-${default_data_config}}
network_config=${NETWORK_CONFIG:-${default_network_config}}
run_name=${RUN_NAME:-baseline}
model_prefix=${MODEL_PREFIX:-result/stxs/${run_name}/fullmodel}
batch_size=${BATCH:-512}
gpu=${GPU-0}

[[ -f ${data_config} ]] || { echo "Data config not found: ${data_config}" >&2; exit 1; }
[[ -f ${network_config} ]] || { echo "Model config not found: ${network_config}" >&2; exit 1; }
compgen -G "${data_glob}" > /dev/null || { echo "No input files match DATA_GLOB=${data_glob}" >&2; exit 1; }

mkdir -p "$(dirname "${model_prefix}")" logs_stxs

# TARGET_CENTER (regression only): the label becomes  target_logpt = log(pT) - TARGET_CENTER.
# The launcher writes a copy of the data config with this center into the run folder and uses that copy, and
# passes the same number to the loss (-o target_center). The value is stored in the run folder at training, so
# MODE=test uses it automatically. Without TARGET_CENTER the label is log(pT), as before.
center_args=()
if [[ "${regression}" == "1" ]]; then
    center_file="$(dirname "${model_prefix}")/target_center.txt"
    target_center=${TARGET_CENTER:-}
    if [[ "${mode}" == "test" && -f "${center_file}" ]]; then
        trained_center=$(cat "${center_file}")
        if [[ -n "${target_center}" && "${target_center}" != "${trained_center}" ]]; then
            echo "TARGET_CENTER=${target_center} but the model was trained with ${trained_center}" >&2
            exit 1
        fi
        target_center=${trained_center}
    fi
    if [[ -n "${target_center}" ]]; then
        centered_config="$(dirname "${model_prefix}")/$(basename "${data_config}" .yaml)_center${target_center}.yaml"
        python - "${data_config}" "${centered_config}" "${target_center}" <<'PYEOF'
import sys
source, destination, center = sys.argv[1], sys.argv[2], sys.argv[3]
float(center)  # stops here if TARGET_CENTER is not a number
label = "target_logpt: np.log(np.maximum(GenHiggs_pt, 1.))"
text = open(source).read()
if text.count(label) != 1:
    sys.exit(f"Expected the line '{label}' exactly once in {source}")
open(destination, "w").write(text.replace(label, f"{label} - {center}"))
PYEOF
        data_config=${centered_config}
        if [[ "${mode}" == "train" ]]; then echo "${target_center}" > "${center_file}"; fi
        center_args=(-o target_center "${target_center}")
        echo "Regression target: log(pT) - ${target_center}, data config ${data_config}"
    elif [[ "${mode}" == "train" ]]; then
        rm -f "${center_file}"  # this run is trained without a center
    fi
fi

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
            ${regression_args[@]+"${regression_args[@]}"} \
            ${center_args[@]+"${center_args[@]}"} \
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
            ${regression_args[@]+"${regression_args[@]}"} \
            ${center_args[@]+"${center_args[@]}"} \
            "$@"
        ;;
    *)
        echo "MODE must be train or test" >&2
        exit 1
        ;;
esac