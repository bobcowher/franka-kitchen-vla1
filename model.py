"""A VLM with an action head bolted on.

The policy is SmolVLM2-500M reading one camera frame and one instruction, with
a Linear(960 -> 9) on the end that emits joint velocity. That linear layer is
the entire "action capability"; everything else is a pretrained VLM, frozen.

Why the head reads the last token and nothing else: the text stack is causal,
so the final prefix token has already attended to every image token and every
instruction token. For one action per forward pass a learned query token would
be redundant with it. Chunking is where that stops being true -- k actions need
k query positions, because one hidden state cannot carry k distinct answers --
and that is the point at which the inputs_embeds path below becomes necessary.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText

from tasks import TASK_DESCRIPTIONS

VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# vision_config.image_size. Shards load at 448 -- an even halving of the 896
# archive -- and this is the one resize in the project that is not integral.
# It is an upsample, so it cannot alias and loses nothing the 448 frame had;
# frames.py's integer rule exists to stop backends disagreeing on the way down.
VLM_IMAGE_SIZE = 512

# text_config.hidden_size.
HIDDEN = 960

# The processor's image branch is only resize -> /255 -> (x - mean) / std, with
# mean = std = 0.5. Reproducing it on the GPU is what makes the loop affordable.
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5


class Model(nn.Module):

    def __init__(self, num_actions, checkpoint_dir='checkpoints',
                 name='vla_network'):
        super().__init__()

        processor = AutoProcessor.from_pretrained(VLM)
        # 17 sub-images (a 4x4 grid plus a thumbnail) is a document-reading
        # default. Off, one frame is 79-84 tokens instead of 1139.
        processor.image_processor.do_image_splitting = False

        # The prompt is a pure function of the task and there are seven tasks,
        # so tokenization leaves the training loop entirely. Nothing the VLM
        # produces is cached -- only the tokens -- so unfreezing it later
        # invalidates none of this.
        self._build_prompts(processor)

        # .model drops the language-modelling head: 49280 x 960 of vocabulary
        # projection that nothing downstream reads.
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            VLM, dtype=torch.bfloat16, attn_implementation="sdpa").model
        self.vlm.requires_grad_(False)

        # The LayerNorm is not decoration. Measured on ten real frames, the last
        # hidden state is 95% constant: ||mean|| 53.8 against ||deviation|| 2.8,
        # with dim 232 alone carrying |mean| 34.0, sixty times the median dim.
        # That is SmolLM2's massive-activation outliers, and a bare Linear reading
        # them stalls -- 0.0075 after 400 steps where this reaches 0.000000.
        # Centering and rescaling per sample is what puts the 5% that varies with
        # the image on the same footing as the 95% that never does.
        #
        # No tanh. The BC stack had one, and with +/-1 gripper targets it makes
        # zero loss unreachable -- which would blunt the only unambiguous test
        # this design has, overfitting a handful of samples until loss vanishes.
        # The env clips to the joint velocity bounds anyway (_ctrl_velocity_limits).
        self.head = nn.Sequential(nn.LayerNorm(HIDDEN),
                                  nn.Linear(HIDDEN, num_actions))

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.train()

    def _build_prompts(self, processor):
        """Tokenize the seven instructions once, padded to a common length.

        The instructions differ in length (79 to 84 tokens), so a batch mixing
        tasks has to pad. Padding goes on the right and the head gathers each
        row's last real token: under causal attention a real token cannot see a
        later pad, so the gathered state is identical to the unpadded one. Left
        padding would be the trap -- it shifts every RoPE position.
        """
        prompts = []
        for description in TASK_DESCRIPTIONS:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": description},
            ]}]
            prompts.append(processor.apply_chat_template(
                messages, add_generation_prompt=True))

        # One dummy frame per prompt: the processor needs an image to know how
        # many image tokens to expand, and resizes whatever it is given.
        dummy = np.zeros((VLM_IMAGE_SIZE, VLM_IMAGE_SIZE, 3), dtype=np.uint8)
        tokens = processor(text=prompts, images=[[dummy]] * len(prompts),
                           padding=True, return_tensors="pt")

        self.register_buffer("prompt_ids", tokens["input_ids"], persistent=False)
        self.register_buffer("prompt_mask", tokens["attention_mask"],
                             persistent=False)
        self.register_buffer("prompt_end", tokens["attention_mask"].sum(1) - 1,
                             persistent=False)

    def train(self, mode=True):
        """Keep the VLM in eval even while the head trains.

        nn.Module.train() recurses, so without this the frozen tower's dropout
        switches on for training and off for rollout -- a difference between the
        two paths that nothing would report.
        """
        super().train(mode)
        self.vlm.eval()
        return self

    def preprocess(self, frames):
        """uint8 (B,H,W,3) or (H,W,3) -> pixel_values (B,1,3,512,512).

        The one place a frame becomes model input, so training and rollout
        cannot drift apart. Running this on the GPU rather than calling the
        processor is the difference between 1.0 and 7.0 iterations/sec at batch
        64 -- the processor's CPU image path was 85% of the step.
        """
        if not torch.is_tensor(frames):
            frames = torch.from_numpy(np.ascontiguousarray(frames))
        if frames.dim() == 3:
            frames = frames[None]

        x = frames.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255
        x = nn.functional.interpolate(x, size=VLM_IMAGE_SIZE, mode="bilinear",
                                      align_corners=False)
        x = (x - IMAGE_MEAN) / IMAGE_STD
        return x.to(torch.bfloat16).unsqueeze(1)

    def forward(self, frames, task):
        task = task.reshape(-1).to(self.device)
        pixel_values = self.preprocess(frames)

        out = self.vlm(input_ids=self.prompt_ids[task],
                       attention_mask=self.prompt_mask[task],
                       pixel_values=pixel_values)

        last = out.last_hidden_state[torch.arange(len(task), device=self.device),
                                     self.prompt_end[task]]
        return self.head(last.float())

    @property
    def device(self):
        return next(self.head.parameters()).device

    def save_checkpoint(self):
        """Only the head. The other 507M parameters are frozen and already on
        disk in the HF cache; writing them every save costs a GB for nothing."""
        torch.save(self.head.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.head.load_state_dict(torch.load(self.checkpoint_file))
