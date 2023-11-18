from typing import Tuple, Union
import dgl
import torch
from dgl.nn import AvgPooling
from torch import nn
from settings import GatedGATConfig, GatedGraphAttentionConv,CGGATConv
from settings import RBFExpansion, embbedding


class CGGAT(nn.Module):

    def __init__(self, config: GatedGATConfig = GatedGATConfig(name="gategat")):
        super().__init__()
        """embedding layer"""
        self.atom_embedding = embbedding(
            config.atom_input_features, config.hidden_features
        )
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features,),
            embbedding(config.edge_input_features, config.embedding_features),
            embbedding(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.angle_input_features,
            ),
            embbedding(config.angle_input_features, config.embedding_features),
            embbedding(config.embedding_features, config.hidden_features),
        )
        """Convolution layer"""
        self.gategcn_layers = nn.ModuleList(
            [CGGATConv(config.hidden_features, config.hidden_features)
                for idx in range(config.gategcn_layers)])

        self.atomgcn_layers = GatedGraphAttentionConv(config.hidden_features, config.hidden_features)

        # readout layer
        self.readout = AvgPooling()
        # self.fc1 = nn.Linear(config.hidden_features, config.hidden_features)
        self.fc = nn.Linear(config.hidden_features, config.output_features)

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        # v: atom features (g.ndata)
        # e: bond features (g.edata and lg.ndata)
        # a: angle features (lg.edata)
        g, Eg = g
        Eg = Eg.local_var()
        # angle features
        a = self.angle_embedding(Eg.edata.pop("h"))
        g = g.local_var()
        # initial node features: atom feature
        v = g.ndata.pop("atom_features")
        v = self.atom_embedding(v)
        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        e = self.edge_embedding(bondlength)
        #  update node, edge, angle features
        for gategcn_layer in self.gategcn_layers:
            v, e, a = gategcn_layer(g, Eg, v, e, a)

        v, e = self.atomgcn_layers(g, v, e)
        # norm-activation-pool
        h = self.readout(g, v)
        # h = self.fc1(torch.relu(h))
        out = self.fc(h)

        return torch.squeeze(out)  # h

