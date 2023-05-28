from torch import nn

class NonLinear(nn.Module):
    """Create A Non-Linear Model"""
    def __init__(self, out_features, output_dim:int=1) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(nn.LazyLinear(out_features),
                                nn.LeakyReLU(),
                                nn.Linear(out_features, output_dim)
        )

        self.name=f'NonLinear_{out_features}'
    
    def forward(self, x):
        x = self.fc1(x)

        return x



class NARX(nn.Module):
    """This module is setup as a NARX model"""
    def __init__(self, out_features, output_dim:int=1):
        super().__init__()
        self.fc1 = nn.Sequential(nn.LazyLinear(out_features),
                                nn.LeakyReLU(),
                                nn.Linear(out_features, output_dim),
        )

        self.name = f'NARX_{out_features}'
        
    def forward(self, x):
        x = self.fc1(x)

        return x
