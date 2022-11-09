from torch import nn
import torch.nn.functional as F
import torch as T
import numpy as np

GAMMA=0.99
device = 'cuda' if T.cuda.is_available() else 'cpu'

def nature_cnn(obs, depths=(32, 64, 64), final_layer = 512):
    n_input_channels = obs.shape[0]
    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Flatten()
    )
    with T.no_grad():
        n_flatten = cnn(T.as_tensor(obs.sample()[None]).float()).shape[1]
    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())
    return out

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.net = nn.Sequential(
            nature_cnn(env.observation_space),
            nn.Linear(512, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = T.as_tensor(obs, dtype=T.float32).to(device)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = T.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

    def train_loss(self, transitions, target_net):
        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = T.as_tensor(obses, dtype=T.float32).to(device)
        actions_t = T.as_tensor(actions, dtype=T.int64).unsqueeze(-1).to(device)
        rews_t = T.as_tensor(rews, dtype=T.float32).unsqueeze(-1).to(device)
        dones_t = T.as_tensor(dones, dtype=T.float32).unsqueeze(-1).to(device)
        new_obses_t = T.as_tensor(new_obses, dtype=T.float32).to(device)

        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA*(1-dones_t)*max_target_q_values

        q_values = self(obses_t)
        action_q_values = T.gather(q_values, dim=1, index=actions_t)

        loss = F.smooth_l1_loss(action_q_values, targets)
        return loss
        