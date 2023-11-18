import dgl
import dgl.function as fn
import numpy as np
import torch
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F
from dgl.nn.pytorch import GATConv, GATv2Conv
from typing import Optional
from pydantic import BaseSettings as PydanticBaseSettings


class BaseSettings(PydanticBaseSettings):

    class Config:
        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"


class GatedGATConfig(BaseSettings):

    name: Literal["gategat"]
    gategcn_layers: int = 4
    gcn_layers: int = 1
    atom_input_features: int = 92
    edge_input_features: int = 40
    angle_input_features: int = 20
    embedding_features: int = 64
    hidden_features: int = 128
    output_features: int = 1

    class Config:
        # Configure model settings
        env_prefix = "jv_model"


# RBF expansion
class RBFExpansion(nn.Module):
    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )
        if lengthscale is None:
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )


class GatedGraphAttentionConv(nn.Module):
    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.gat_layers = GATConv(in_feats=input_features,
                                  out_feats=output_features,
                                  allow_zero_in_degree=True,
                                  num_heads=2)
        self.bn_edges = nn.BatchNorm1d(output_features)
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        g = g.local_var()
        # edge update
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)
        # node update
        g.ndata['node_feats'] = self.src_update(node_feats)
        update_node_features = self.gat_layers(g, g.ndata['node_feats']).flatten(1)
        # x = update_node_features + self.src_update(node_feats)# + m / edge_feats.shape[0]

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = g.ndata.pop("h") + update_node_features
        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y
        return x, y


class CGGATConv(nn.Module):
    def __init__(
        self, in_features: int, out_features: int,
    ):
        super().__init__()
        self.node_update = GatedGraphAttentionConv(in_features, out_features)
        self.edge_update = GatedGraphAttentionConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        Eg: dgl.DGLGraph,
        v: torch.Tensor,
        e: torch.Tensor,
        a: torch.Tensor,
    ):
        # Node ,Edge, Angle updates
        # v: node input features
        # e: edge input features
        # a: angle input features

        g = g.local_var()
        Eg = Eg.local_var()
        # update on atom graph
        v, m = self.node_update(g, v, e)
        # update on edge graph
        e, z = self.edge_update(Eg, m, a)
        return v, e, a


class embbedding(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.fc(x)
