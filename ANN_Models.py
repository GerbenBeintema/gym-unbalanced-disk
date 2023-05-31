from torch import nn, zeros, stack, cat, float64

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
    
class RNN(nn.Module):
    """This module is setup as a RNN model"""
    def __init__(self, input_size:int=1, hidden_size:int=40, output_size:int=1, nr_nodes:int=40):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nr_nodes = nr_nodes
        self.output_size = output_size

        net = lambda n_in,n_out: nn.Sequential(nn.Linear(n_in, self.nr_nodes),
                                               nn.LeakyReLU(),
                                               nn.Linear(self.nr_nodes, n_out)
        ).double()

        # Initialize the network
        self.H2H = net(self.input_size + self.hidden_size, self.hidden_size)
        self.H2O = net(self.input_size + self.hidden_size, self.output_size)

        self.name = f'RNN'
        
    def forward(self, inputs):
        """forward pass of the RNN model"""

        # Initialize hidden state
        hidden = zeros(inputs.size(0), self.hidden_size, dtype=float64, device=inputs.device)
        outputs = []

        for i in range(inputs.size(1)):
            # Set up data for this timestep
            u = inputs[:, i]
            combined = cat((u[:, None], hidden), dim=1)

            # Update hidden state
            hidden = self.H2H(combined)
            outputs.append(self.H2O(combined)[:,0])

        return stack(outputs, dim=1)