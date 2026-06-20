import numpy as np
from veritas_core.graph import generate_network, NODE_FEATURE_DIM, Subgraph
from veritas_core import gnn
from veritas_core.aggregation import fedavg, multi_krum
from veritas_core.dp import privatize


def _net_and_sg(seed=0):
    net = generate_network(n_banks=4, n_accounts=500, n_campaigns=8, seed=seed)
    return net, net.local_subgraph(0)


def test_predict_proba_in_range():
    _, sg = _net_and_sg()
    w = gnn.init_weights(NODE_FEATURE_DIM, hidden=8, seed=1)
    p = gnn.predict_proba(w, sg)
    assert p.shape[0] == sg.n_nodes
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_train_local_improves_recall():
    net = generate_network(n_banks=4, n_accounts=600, n_campaigns=10, seed=11)
    sg = net.local_subgraph(0)
    w0 = gnn.init_weights(NODE_FEATURE_DIM, hidden=8, seed=2)
    r0 = gnn.recall(w0, sg)
    w1 = gnn.train_local(w0, sg, epochs=200, lr=0.15)
    r1 = gnn.recall(w1, sg)
    assert r1 > r0


def test_message_passing_uses_neighbours():
    """A node's output must change when its NEIGHBOURHOOD changes."""
    _, sg = _net_and_sg(seed=21)
    w = gnn.init_weights(NODE_FEATURE_DIM, hidden=8, seed=3)
    p_before = gnn.predict_proba(w, sg)

    # find a node that has at least one neighbour
    deg = sg.A.sum(axis=1)
    node = int(np.argmax(deg))
    assert deg[node] > 0

    # build a perturbed subgraph: remove node's edges (isolate it)
    A2 = sg.A.copy()
    A2[node, :] = 0.0
    A2[:, node] = 0.0
    sg2 = Subgraph(sg.X, sg.y, A2, sg.owned_mask, sg.global_ids)
    p_after = gnn.predict_proba(w, sg2)

    # the isolated node's probability must change because aggregation no longer
    # mixes in neighbour features
    assert not np.isclose(p_before[node], p_after[node])


def test_weight_vector_fixed_dim_across_banks():
    net = generate_network(n_banks=4, n_accounts=500, n_campaigns=8, seed=31)
    ws = []
    for b in range(net.n_banks):
        sg = net.local_subgraph(b)
        w = gnn.init_weights(NODE_FEATURE_DIM, hidden=8, seed=4)
        w = gnn.train_local(w, sg, epochs=5, lr=0.1)
        ws.append(w)
    dims = {len(w) for w in ws}
    assert len(dims) == 1  # identical dimension regardless of subgraph size
    assert dims.pop() == gnn.weight_dim(NODE_FEATURE_DIM, 8)

    # fedavg / multi_krum / dp accept the list of flat vectors unchanged
    avg = fedavg(ws)
    assert avg.shape == ws[0].shape
    agg, sel = multi_krum(ws, n_byzantine=0, m=net.n_banks)
    assert agg.shape == ws[0].shape
    rng = np.random.default_rng(0)
    priv = privatize(ws[0] - ws[1], 4.0, 0.01, rng)
    assert priv.shape == ws[0].shape


def test_pack_unpack_roundtrip():
    w = gnn.init_weights(NODE_FEATURE_DIM, hidden=8, seed=5)
    W1, b1, W2, b2 = gnn._unpack(w, NODE_FEATURE_DIM, 8)
    w2 = gnn._pack(W1, b1, W2, b2)
    assert np.allclose(w, w2)
    assert len(w) == gnn.weight_dim(NODE_FEATURE_DIM, 8)
