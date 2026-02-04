from torch import nn 


class LinearRegression(nn.Module):
    def __init__(self, input_dim, bias = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=bias )

    def forward(self, X):
        return self.linear(X)
