from torch import nn

class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Sequential(nn.LazyLinear(5),
                                nn.LeakyReLU(),
                                nn.Linear(5, output_dim),
        )

        self.name = 'ANN'
        
    def forward(self, x):
        x = self.fc1(x)

        return x