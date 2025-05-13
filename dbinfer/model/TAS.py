import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.functional import edge_softmax

# TAS code init commit for the repo
# Test Version

# Compute the score of each attribute by TAS module
class TabularAttributeSelector(nn.Module):
    """
    TAS module using DGL for bipartite attention from row nodes to attribute nodes.

    Args:
        in_dim (int): Input feature dimension for both row and attribute nodes.
        proj_dim (int): Dimension of query/key projections (d_k).
    """
    def __init__(self, in_dim: int, proj_dim: int):
        super(TabularAttributeSelector, self).__init__()
        self.W_q = nn.Linear(in_dim, proj_dim, bias=False)
        self.W_k = nn.Linear(in_dim, proj_dim, bias=False)
        self.W_v = nn.Linear(in_dim, proj_dim, bias=False)
        self.sqrt_dk = proj_dim ** 0.5

    def forward(self, g: dgl.DGLHeteroGraph, 
                feat_row: torch.Tensor, feat_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation for TAS. Computes a score for each attribute node.

        Args:
            g (DGLHeteroGraph): Heterogeneous graph with edge type ('row','has_attr','attr').
            feat_row (Tensor): Row node features, shape (n_rows, in_dim).
            feat_attr (Tensor): Attribute node features, shape (n_attrs, in_dim).

        Returns:
            Tensor: Normalized importance scores for each attribute, shape (n_attrs,).
        """
        g.nodes['row'].data['h'] = feat_row
        g.nodes['attr'].data['h'] = feat_attr

        g.nodes['row'].data['q'] = self.W_q(feat_row)
        g.nodes['attr'].data['k'] = self.W_k(feat_attr)
        # Value if needed for downstream message passing
        g.nodes['row'].data['v'] = self.W_v(feat_row)
        g.nodes['attr'].data['v'] = self.W_v(feat_attr)

        # Compute Score
        def compute_score(edges):
            score = (edges.src['q'] * edges.dst['k']).sum(dim=-1)
            return {'e': score / self.sqrt_dk}
        g.apply_edges(compute_score, etype='has_attr')

        e = g.edges['has_attr'].data['e']
        alpha = edge_softmax(g, e, edge_type='has_attr')
        g.edges['has_attr'].data['alpha'] = alpha

        weighted = alpha * torch.log1p(alpha)
        g.edges['has_attr'].data['w'] = weighted

        
        g.update_all(
            message_func=dgl.fn.copy_e('w', 'w'),
            reduce_func=dgl.fn.sum('w', 'score'),
            etype='has_attr'
        )
        scores = g.nodes['attr'].data['score']  # shape (n_attrs,)

        S = torch.softmax(scores, dim=0)

        # return the score
        return S

if __name__ == "__main__":

    n_rows, n_attrs = 100, 20
    in_dim, proj_dim = 64, 16

    rows = torch.arange(n_rows).repeat_interleave(3)
    attrs = torch.randint(0, n_attrs, (n_rows * 3,))
    g = dgl.heterograph({
        ('row', 'has_attr', 'attr'): (rows, attrs),
        ('attr', 'rev_has_attr', 'row'): (attrs, rows)
    })

    feat_row = torch.randn(n_rows, in_dim)
    feat_attr = torch.randn(n_attrs, in_dim)

    tas = TabularAttributeSelector(in_dim, proj_dim)
    importance_scores = tas(g, feat_row, feat_attr)

    print("Attribute importance scores:\n", importance_scores) == "__main__":
    n_rows, n_attrs, in_dim, proj_dim = 100, 10, 64, 32
    tas = TabularAttributeSelector(in_dim=in_dim, proj_dim=proj_dim)

    X_row = torch.randn(n_rows, in_dim)
    X_attr = torch.randn(n_attrs, in_dim)

    scores = tas(X_row, X_attr)
    print("Attribute importance scores:", scores)
