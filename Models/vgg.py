import torch
import torch.nn as nn  
import time

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512,  512, "M", 512, 512, 512, "M"]
}


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        x = x.cuda()
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
        )
        self.fcs = nn.Sequential(
            nn.Linear(128*30*30, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.shape[0], -1)
        return self.fcs(x)




def train_step(model, state_transitions, tgt, num_actions):

    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).cuda()
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).cuda()
    rewards = torch.stack(([torch.tensor([s.reward]) for s in state_transitions])).cuda()
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).cuda()
    actions = [s.action for s in state_transitions]

    criterion = torch.optim.Adam(model.parameters(), lr=0.0001)

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0].cuda()
    
    qvals = model(cur_states).cuda()
    one_hot_actions = torch.nn.functional.one_hot(torch.LongTensor(actions), num_actions).cuda()
    criterion.zero_grad()

    loss = ((rewards + mask[:,0]*qvals_next - torch.sum(qvals*one_hot_actions, -1))**2).mean()
    loss.backward()
    criterion.step()

    return loss



if __name__ == "__main__":
    model = VGG_net(in_channels=1, num_classes=3).cuda()
    mod = NeuralNet(3).cuda()
    x = torch.randn(1, 1, 128, 128).cuda()
    t1 = time.time()
    print(model(x).shape)
    t2 = time.time()
    t2 -= t1
    print(t2)
    t1 = time.time()
    print(mod(x).shape)
    t2 = time.time()
    t2 -= t1
    print(t2)