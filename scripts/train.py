import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

# The shards live outside the repo -- a symlink locally, a real path on the
# training server -- so the location is environment, not code.
data_path = os.environ.get("DATASET_PATH", "dataset")

agent = Agent(data_path=data_path, name="vla_network")

agent.train(epochs=100001, batch_size=64)
