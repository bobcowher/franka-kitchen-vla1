import os
import sys

# lab's 3090 is GPU 1 to nvidia-smi but GPU 0 to CUDA, which orders by speed
# unless told otherwise. Beekeeper exports the nvidia-smi index, so it lands on
# the 3060; pinning the order makes the two agree.
if os.environ.get("BEEKEEPER_RUN_DIR"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

agent = Agent(data_path=os.environ.get("DATASET_PATH", "dataset"),
              name="vla_network")

agent.train(epochs=100001, batch_size=64)
