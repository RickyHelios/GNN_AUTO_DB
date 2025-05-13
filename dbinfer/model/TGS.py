def tgs_sampling(g: dgl.DGLHeteroGraph,
                 seeds: torch.Tensor,
                 max_hops: int = 2,
                 max_sample_per_hop: int = 5) -> list:
    """
    Node-wise Tabular Graph Sampling (TGS) on a heterograph.
    Args:
        g: DGLHeteroGraph with ('row','has_attr','attr') edges.
        seeds: initial row node IDs (Tensor of ints).
        max_hops: number of sampling hops.
        max_sample_per_hop: max neighbors per node to sample.
    Returns:
        List of sampled node IDs (both row and attribute IDs).
    """
    sampled = set(seeds.tolist())
    frontier = seeds.tolist()

    for h in range(max_hops):
        next_frontier = []
        for v in frontier:
            # 1) Degree-1 chain expansion (row → attr → row → ...)
            u = v
            while True:
                nbrs = g.successors(u, etype='has_attr').tolist()
                # Only continue if exactly one incoming and one outgoing edge
                if len(nbrs) == 1 and len(g.predecessors(u, etype='rev_has_attr')) == 1:
                    new_u = nbrs[0]
                    if new_u not in sampled:
                        sampled.add(new_u)
                        u = new_u
                        continue
                break

            # 2) Sample neighbors of the chain endpoint u
            nbrs = g.successors(u, etype='has_attr').tolist()
            # Filter out already-sampled nodes
            candidates = [w for w in nbrs if w not in sampled]
            # Deterministic or random sampling: here we take the first k
            selected = candidates[:max_sample_per_hop]
            for w in selected:
                sampled.add(w)
                next_frontier.append(w)

        frontier = next_frontier

    return list(sampled)

# Example usage at bottom of TAS.py
if __name__ == "__main__":
    # ... (after building g and computing TAS scores) ...
    seeds = torch.tensor([0, 1])
    sampled_nodes = tgs_sampling(g, seeds, max_hops=3, max_sample_per_hop=4)
    print("TGS sampled node IDs:", sampled_nodes)
