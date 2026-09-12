import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, image_input_shape, 
                 joint_pos_dim, 
                 task_dim,
                 num_actions, 
                 hidden_dim,
                 compression_dim=None, 
                 n_hidden_layers=1,
                 checkpoint_dir='checkpoints', 
                 name='bc_network'):
        super(Model, self).__init__()

        # Per-modality embedding width before fusion. Defaults to hidden_dim
        # (the historical behavior); pass a value to tune it independently.
        if compression_dim is None:
            compression_dim = hidden_dim

        # Six stride-2 stages, halving the frame each time and doubling channels
        # on the way down: 448 -> 224 -> 112 -> 56 -> 28 -> 14 -> 7.
        #
        # The previous stack was Nature-DQN's, built for 84x84. Its kernels and
        # strides are fixed, so feeding it 448 left the receptive field at 36x36
        # -- 18% of an 84px frame but 0.6% of this one, about the size of the
        # gripper fingertips. No conv feature could see the arm and its target at
        # once, and it stopped downsampling at 52x52, so the flatten that follows
        # was Linear(173056, 756): 130.8M params, 97.6% of the whole model, doing
        # the spatial reasoning the convs never did. Going deeper takes the
        # receptive field to 131x131 and the grid to 7x7, which shrinks that
        # layer ~7x and moves capacity into the part that actually looks.
        #
        # GroupNorm rather than BatchNorm: rollouts run this at batch size 1 with
        # .eval(), which is where BatchNorm's running statistics bite.
        channels = [image_input_shape[0], 32, 64, 128, 256, 256, 512]
        stages = []
        for i, (c_in, c_out) in enumerate(zip(channels, channels[1:])):
            # A wider kernel on the stem only; 448px of raw pixels is more than a
            # 3x3 can usefully summarize, and it is cheap at 3 input channels.
            kernel = 7 if i == 0 else 3
            stages += [nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=2,
                                 padding=kernel // 2),
                       nn.GroupNorm(8, c_out),
                       nn.ReLU()]
        self.conv = nn.Sequential(*stages)

        with torch.no_grad():
            dummy = torch.zeros(1, *image_input_shape)
            flat_size = self._conv_forward(dummy).shape[1]

        # We want the total compression dim to be the joint information, for gut feeling reasons. 
        compression_dim_small = compression_dim // 2

        # Ablation: joint_vel is deliberately absent, the mirror of the
        # no-joint-pos run. Velocity was in here to stand in for history -- with
        # a single frame, it is the only thing saying which way the arm was
        # already moving. This asks whether that is worth its place, given that
        # ALOHA/ACT and pi0 both condition on joint position alone.
        self.joint_pos_input = nn.Linear(joint_pos_dim, compression_dim_small)
        self.task_input      = nn.Embedding(task_dim, compression_dim_small)


        self.image_input = nn.Linear(flat_size, compression_dim)

        self.compression_layer = nn.Linear(compression_dim + (compression_dim_small * 2), hidden_dim)

        # n_hidden_layers hidden FC layers between fusion and output.
        # n_hidden_layers=1 reproduces the original single `linear1`.
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )

        self.output = nn.Linear(hidden_dim, num_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.apply(weights_init_)


    def _conv_forward(self, x):
        return self.conv(x).flatten(1)

    def forward(self, obs, joint_pos, task):
        x_image = self._conv_forward(obs)
        x_image = F.relu(self.image_input(x_image))

        x_joint_pos = F.relu(self.joint_pos_input(joint_pos))
        x_task      = F.relu(self.task_input(task))

        # x_task = self.task_input(task_id)

        x = torch.cat([x_image, x_joint_pos, x_task], dim=1)

        x = F.relu(self.compression_layer(x))

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        x = F.tanh(self.output(x))
        # # x = F.tanh(self.out)
        # x = self.output(x)
        # # x = F.tanh(x)
        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

