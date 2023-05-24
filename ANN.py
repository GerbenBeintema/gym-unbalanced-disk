from torch import nn

class ANN(nn.Module):
    """This module is setup as a NARX model"""
    def __init__(self, out_features ,input_dim:int=30, output_dim:int=1):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, out_features),
                                nn.LeakyReLU(),
                                nn.Linear(out_features, output_dim),
        )

        self.name = 'ANN'
        
    def forward(self, x):
        x = self.fc1(x)

        return x

class NARX(nn.Module):
    def __init__(self, input_dim, **kwargs) -> None:
        super().__init__(*args, **kwargs)