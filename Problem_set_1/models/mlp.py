from torch import nn



class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu_stack = nn.Sequential(
            nn.Linear((28*28), 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.relu_stack(x)

