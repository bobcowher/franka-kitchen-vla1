"""Frozen SmolVLM2 with an action head that emits joint velocity."""
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText

from tasks import TASK_DESCRIPTIONS

VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_IMAGE_SIZE = 512   # vision_config.image_size
HIDDEN = 960           # text_config.hidden_size


class ActionHead(nn.Module):
    """Separate norms because the two streams have very different scales."""

    def __init__(self, hidden, num_actions):
        super().__init__()
        self.norm_last = nn.LayerNorm(hidden)
        self.norm_pooled = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden * 2, num_actions)

    def forward(self, last, pooled):
        return self.out(torch.cat([self.norm_last(last),
                                   self.norm_pooled(pooled)], dim=1))


class Model(nn.Module):

    def __init__(self, num_actions, checkpoint_dir='checkpoints',
                 name='vla_network'):
        super().__init__()

        processor = AutoProcessor.from_pretrained(VLM)
        # On, a frame is 17 sub-images and 1139 tokens. Off, 79-84.
        processor.image_processor.do_image_splitting = False
        self._build_prompts(processor)

        # .model drops the unused 49280 x 960 vocabulary projection.
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            VLM, dtype=torch.bfloat16, attn_implementation="sdpa").model
        self.vlm.requires_grad_(False)

        self.head = ActionHead(HIDDEN, num_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.train()

    def _build_prompts(self, processor):
        """Tokenize the seven instructions once; they never change."""
        prompts = []
        for description in TASK_DESCRIPTIONS:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": description},
            ]}]
            prompts.append(processor.apply_chat_template(
                messages, add_generation_prompt=True))

        dummy = np.zeros((VLM_IMAGE_SIZE, VLM_IMAGE_SIZE, 3), dtype=np.uint8)
        tokens = processor(text=prompts, images=[[dummy]] * len(prompts),
                           padding=True, return_tensors="pt")

        self.register_buffer("prompt_ids", tokens["input_ids"], persistent=False)
        self.register_buffer("prompt_mask", tokens["attention_mask"],
                             persistent=False)
        # Right-padded, so the last real token is at mask.sum() - 1. Left
        # padding would shift every RoPE position.
        self.register_buffer("prompt_end", tokens["attention_mask"].sum(1) - 1,
                             persistent=False)
        self.register_buffer(
            "image_positions",
            (tokens["input_ids"] == processor.tokenizer.convert_tokens_to_ids(
                "<image>")).unsqueeze(-1), persistent=False)

    def train(self, mode=True):
        # train() recurses, so the frozen tower needs pinning back to eval.
        super().train(mode)
        self.vlm.eval()
        return self

    def preprocess(self, frames):
        """uint8 (B,H,W,3) -> pixel_values. Same path for training and rollout.

        This is the processor's image branch, on the GPU. Calling the processor
        instead costs 85% of the step.
        """
        if not torch.is_tensor(frames):
            frames = torch.from_numpy(np.ascontiguousarray(frames))
        if frames.dim() == 3:
            frames = frames[None]

        x = frames.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255
        x = nn.functional.interpolate(x, size=VLM_IMAGE_SIZE, mode="bilinear",
                                      align_corners=False)
        return ((x - 0.5) / 0.5).to(torch.bfloat16).unsqueeze(1)

    def forward(self, frames, task):
        task = task.reshape(-1).to(self.device)

        h = self.vlm(input_ids=self.prompt_ids[task],
                     attention_mask=self.prompt_mask[task],
                     pixel_values=self.preprocess(frames)).last_hidden_state.float()

        # The image precedes the text and attention is causal, so image
        # positions cannot see the instruction and the last token can. Need both.
        last = h[torch.arange(len(task), device=self.device), self.prompt_end[task]]
        mask = self.image_positions[task]
        pooled = (h * mask).sum(1) / mask.sum(1)

        return self.head(last, pooled)

    @property
    def device(self):
        return next(self.head.parameters()).device

    def save_checkpoint(self):
        torch.save(self.head.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.head.load_state_dict(torch.load(self.checkpoint_file))
