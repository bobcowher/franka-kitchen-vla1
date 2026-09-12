"""Encode one sentence with SmolVLM2's text tower, the way SmolVLA does it.

lerobot's SmolVLMWithExpert truncates text_model.layers to num_vlm_layers (16 in
lerobot/smolvla_base) and applies text_model.norm after the loop. So the policy
reads norm(h16). That is NOT hidden_states[16], which transformers captures
before the final norm -- the two are near-orthogonal.

sdpa rather than flash_attention_2: flash-attn's setup.py imports torch and pip
builds it isolated, so it cannot install from requirements.txt.
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
NUM_VLM_LAYERS = 16
SENTENCE = "Open the microwave door"

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()

text = model.model.text_model
text.layers = text.layers[:NUM_VLM_LAYERS]

enc = processor.tokenizer(SENTENCE, return_tensors="pt").to("cuda")
with torch.no_grad():
    h = text(**enc).last_hidden_state

print(f"{SENTENCE!r} -> {tuple(enc['input_ids'].shape)} tokens")
print(f"layer {NUM_VLM_LAYERS} hidden state: {tuple(h.shape)}\n")
print(h)
