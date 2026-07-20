import torch.nn as nn
from torch_geometric.nn import GCNConv

class MolecularGNN(nn.Module):
    """ Graph Neural Network for molecular structure encoding.
    
    stacked GCNConv layers to learn atom-level representations
    from node features and band connectivity (edge_index).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(MolecularGNN, self).__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
    
        # Hidden layer
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        #Final Layer    
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.norms.append(nn.LayerNorm(out_channels))

        self.activation = nn.SiLU()

    def forward(self, x, edge_index):
        """
            Args:
                x: Node feature matrix [num_models, in_channels]
                edge_index: Graph connectivity [2, num_edges]

            Returns:
                Node Embeddings [num_nodes, out_channels]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = self.activation(x)

        return x 