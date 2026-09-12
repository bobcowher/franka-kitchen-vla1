#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate farama-kitchen-bc

# The "Loading weights: 0/489" bar counts tensors read off local disk, not bytes
# downloaded, but it is indistinguishable from a download bar.
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Offline mode skips the HEAD-per-file staleness check from_pretrained does on
# every call (~2.5s). It does not fall back to downloading -- an uncached model
# is a hard OSError -- so only set it once the weights are actually on disk.
SMOLVLM_SNAPSHOTS=~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots
[ -d "$SMOLVLM_SNAPSHOTS" ] && export HF_HUB_OFFLINE=1

# Run once
# python $(python -c "import robosuite, os; print(os.path.dirname(robosuite.__file__))")/scripts/setup_macros.py
# export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning"
# export ROBOSUITE_LOG_LEVEL=ERROR   # robosuite uses its own logger, not warnings

#pip install -r requirements.txt

# The main script: training via scripts/train.py. Watch one rollout with
# ./scripts/test.py, collect demos with ./scripts/human_control.py.
# python -u ./scripts/human_control.py
# python -u scripts/train.py
python -u smolvlm_test.py
