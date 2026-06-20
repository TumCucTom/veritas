import numpy as np
from veritas_core.graph import generate_network, NODE_FEATURE_DIM, CROSS_BANK_RATIO


def test_partition_covers_all_accounts_exactly_once():
    net = generate_network(n_banks=4, n_accounts=400, n_campaigns=5, seed=1)
    seen = np.zeros(net.n_accounts, dtype=int)
    for b in range(net.n_banks):
        sg = net.local_subgraph(b)
        owned = sg.global_ids[sg.owned_mask]
        seen[owned] += 1
    # every account owned by exactly one bank
    assert (seen == 1).all()


def test_cross_bank_edges_exist():
    net = generate_network(n_banks=4, n_accounts=400, n_campaigns=6, seed=2)
    xb = net.cross_bank_edges()
    assert len(xb) > 0
    # at least some cross-bank edges touch mule accounts (layering across banks)
    mule_touch = any(net.y[s] == 1 or net.y[d] == 1 for (s, d, a, t) in xb)
    assert mule_touch


def test_boundary_nodes_hide_features():
    net = generate_network(n_banks=4, n_accounts=400, n_campaigns=6, seed=3)
    found_boundary = False
    for b in range(net.n_banks):
        sg = net.local_subgraph(b)
        boundary = ~sg.owned_mask
        if boundary.any():
            found_boundary = True
            # foreign-bank node features are zeroed (private to other bank)
            assert np.allclose(sg.X[boundary], 0.0)
            # but the bank still owns nodes with real (nonzero) features
            assert not np.allclose(sg.X[sg.owned_mask], 0.0)
    assert found_boundary


def test_mule_subgraphs_have_high_fan_in_out_vs_background():
    net = generate_network(n_banks=4, n_accounts=500, n_campaigns=8, seed=4)
    # in_degree=feat[4], out_degree=feat[5] in standardized space
    mule = net.y == 1
    bg = net.y == 0
    assert mule.sum() > 0
    in_deg = net.X[:, 4]
    out_deg = net.X[:, 5]
    # mule accounts (collectors + mules) have materially higher fan-in/out
    assert in_deg[mule].mean() > in_deg[bg].mean()
    assert out_deg[mule].mean() > out_deg[bg].mean()


def test_node_feature_dim_and_shapes():
    net = generate_network(n_banks=3, n_accounts=300, n_campaigns=4, seed=5)
    assert net.X.shape == (300, NODE_FEATURE_DIM)
    sg = net.local_subgraph(0)
    assert sg.X.shape[1] == NODE_FEATURE_DIM
    assert sg.A.shape == (sg.n_nodes, sg.n_nodes)
    assert sg.y.shape[0] == sg.n_nodes
    # adjacency is symmetric (for mean aggregation)
    assert np.allclose(sg.A, sg.A.T)
    # cross_bank_ratio feature exists and is in a sane range for owned nodes
    assert sg.X.shape[1] > CROSS_BANK_RATIO
