import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, HeteroConv, global_mean_pool

class RSSITransformerModel(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, embed_dim=32, num_ap=175, use_residual=True, heads=6, dropout=0.3):
        super().__init__()
        self.use_residual = use_residual
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        # AP 节点embedding
        self.ap_embed = nn.Embedding(num_ap, embed_dim)

        self.project_layers = nn.ModuleList([
            nn.ModuleDict({
                'child': nn.Linear(hidden_dim * heads, hidden_dim),
                'ap': nn.Linear(hidden_dim * heads, hidden_dim)
            }) for _ in range(3)
        ])

        self.convs = nn.ModuleList()
        for layer_idx in range(3):
            if layer_idx == 0:
                input_dim_child = in_channels
                input_dim_ap = embed_dim
            else:
                input_dim_child = hidden_dim
                input_dim_ap = hidden_dim

            
            self.convs.append(HeteroConv({
                ('child', 'sense', 'ap'): TransformerConv((input_dim_child, input_dim_ap), hidden_dim, edge_dim=1, heads=heads, concat=True),
                ('ap', 'rev_sense', 'child'): TransformerConv((input_dim_ap, input_dim_child), hidden_dim, edge_dim=1, heads=heads, concat=True),
                ('child', 'intra', 'child'): TransformerConv((input_dim_child, input_dim_child), hidden_dim, edge_dim=1, heads=heads, concat=True),
            }, aggr='mean'))

        self.norms = nn.ModuleDict({
            'child': nn.LayerNorm(hidden_dim),
            'ap': nn.LayerNorm(hidden_dim)
        })

        
        self.linear_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        device = data['child'].x.device
        
        # solve bugs by mod num_embeddings
        ap_indices = torch.arange(data['ap'].num_nodes, device=device) % self.ap_embed.num_embeddings
        data['ap'].x = self.ap_embed(ap_indices)

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        for i, conv in enumerate(self.convs):
            x_prev = {k: v.clone() for k, v in x_dict.items()} if self.use_residual else None

            x_out = conv(x_dict, edge_index_dict, edge_attr_dict)

            
            for node_type in x_out:
                x_out[node_type] = self.project_layers[i][node_type](x_out[node_type])

            for key in x_out:
                if self.use_residual and key in x_prev and x_out[key].shape == x_prev[key].shape:
                    x_out[key] = x_out[key] + x_prev[key]
                x_out[key] = self.norms[key](x_out[key])
                x_out[key] = self.dropout(x_out[key])

            x_dict = x_out

        child_x = x_dict['child']
        batch_index = data['child'].batch
        graph_repr = global_mean_pool(child_x, batch_index)

        out = self.linear_out(graph_repr)
        return out