import torch

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def create_graph_2hop_recip(ranks, simm, q, k, identities, embeddings):
    nbhs = ranks[q, :k]
    nbhnbhs = ranks[nbhs, :k]

    source_nodes = torch.cat([torch.tensor([q]), nbhs]).repeat_interleave(k)
    dst_nodes = torch.cat([nbhs, nbhnbhs.ravel()])
    edge_index = torch.stack([source_nodes, dst_nodes])
    node_ids, edge_index = torch.unique(edge_index, sorted=True, return_inverse=True)
    remapped_query_node = torch.where(node_ids == q)[0].item()
    # numnodes = node_ids.numel() + 1
    # nodes = torch.arange(0, numnodes, dtype=torch.int64)
    edge_attr = simm[source_nodes, dst_nodes]
    y = torch.tensor(identities[q] == identities[node_ids]).reshape(-1, 1).to(torch.float32)
    edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    # emb = torch.nn.functional.normalize(embeddings, dim=1)
    emb = embeddings
    sim_to_q = emb[node_ids] @ emb[q]  # (N,)
    is_reciprocal = (ranks[node_ids, :k] == q).any(dim=1).to(torch.float32)
    is_reciprocal[node_ids == q] = 1.0
    is_query = (node_ids == q).float()
    # similarity to the own nns?
    # rank_feat = torch.zeros(len(node_ids));
    x = torch.cat([is_reciprocal[:, None], sim_to_q[:, None], is_query[:, None]], dim=1)
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce="max")

    # y = torch.cat([torch.tensor([1]), y]).reshape(-1, 1).to(torch.float32)

    graph = Data(x, edge_index, edge_attr, y, query=remapped_query_node)
    return graph
