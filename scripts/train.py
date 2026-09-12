import os
import sys

# Beekeeper exports CUDA_VISIBLE_DEVICES=0, the 3060, whatever its banner says.
if os.environ.get("BEEKEEPER_RUN_DIR"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

agent = Agent(data_path=os.environ.get("DATASET_PATH", "dataset"),
              name="vla_network")

agent.train(epochs=100001, batch_size=64)
