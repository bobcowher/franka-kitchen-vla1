import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

# agent = Agent(data_path="dataset/microwave", name="bc_microwave")
agent = Agent(data_path="dataset", name="vla_network")

# 7 it/s at batch 64 on the 5090, plus 1.1 min per eval block, so 30K is about
# 1.75 hours. The conv stack first swept all three tasks at 27K.
epochs = 30001

agent.train(epochs=epochs, batch_size=64)
