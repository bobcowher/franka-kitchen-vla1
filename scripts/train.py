import os
import sys

# lab's 3090 is GPU 1 to nvidia-smi but GPU 0 to CUDA, which orders by speed
# unless told otherwise. Beekeeper exports the nvidia-smi index, so it lands on
# the 3060; pinning the order makes the two agree.
if os.environ.get("BEEKEEPER_RUN_DIR"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Experiment (run 10): A/B unfreezing against run 9's frozen-VLM baseline.
# Run 9 hit its best eval/mean (56%) by epoch 40000 and its first non-zero
# signal by 12500, so 20000 epochs is enough to compare against run 9's
# eval/mean at the same checkpoints without paying for a full 100K run.
# Revert both of these (and the epoch count below) once the A/B is read.
os.environ.setdefault("UNFREEZE_LAST_N_LAYERS", "2")
os.environ.setdefault("BACKBONE_LR", "0.0001")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

agent = Agent(data_path=os.environ.get("DATASET_PATH", "dataset"),
              name="vla_network")

agent.train(epochs=int(os.environ.get("VLA_EPOCHS", "20001")), batch_size=64)
