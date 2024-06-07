import dgl.graphbolt as gb
import pytest
import torch


def test_find_reverse_edges_homo():
    edges = (torch.tensor([1, 3, 5]), torch.tensor([2, 4, 5]))
    edges = gb.add_reverse_edges(edges)
    expected_edges = (
        torch.tensor([1, 3, 5, 2, 4, 5]),
        torch.tensor([2, 4, 5, 1, 3, 5]),
    )
    assert torch.equal(edges[0], expected_edges[0])
    assert torch.equal(edges[1], expected_edges[1])


def test_find_reverse_edges_hetero():
    edges = {
        "A:r:B": (torch.tensor([1, 5]), torch.tensor([2, 5])),
        "B:rr:A": (torch.tensor([3]), torch.tensor([3])),
    }
    edges = gb.add_reverse_edges(edges, {"A:r:B": "B:rr:A"})
    expected_edges = {
        "A:r:B": (torch.tensor([1, 5]), torch.tensor([2, 5])),
        "B:rr:A": (torch.tensor([3, 2, 5]), torch.tensor([3, 1, 5])),
    }
    assert torch.equal(edges["A:r:B"][0], expected_edges["A:r:B"][0])
    assert torch.equal(edges["A:r:B"][1], expected_edges["A:r:B"][1])
    assert torch.equal(edges["B:rr:A"][0], expected_edges["B:rr:A"][0])
    assert torch.equal(edges["B:rr:A"][1], expected_edges["B:rr:A"][1])


def test_find_reverse_edges_bi_reverse_types():
    edges = {
        "A:r:B": (torch.tensor([1, 5]), torch.tensor([2, 5])),
        "B:rr:A": (torch.tensor([3]), torch.tensor([3])),
    }
    edges = gb.add_reverse_edges(edges, {"A:r:B": "B:rr:A", "B:rr:A": "A:r:B"})
    expected_edges = {
        "A:r:B": (torch.tensor([1, 5, 3]), torch.tensor([2, 5, 3])),
        "B:rr:A": (torch.tensor([3, 2, 5]), torch.tensor([3, 1, 5])),
    }
    assert torch.equal(edges["A:r:B"][0], expected_edges["A:r:B"][0])
    assert torch.equal(edges["A:r:B"][1], expected_edges["A:r:B"][1])
    assert torch.equal(edges["B:rr:A"][0], expected_edges["B:rr:A"][0])
    assert torch.equal(edges["B:rr:A"][1], expected_edges["B:rr:A"][1])


def test_find_reverse_edges_circual_reverse_types():
    edges = {
        "A:r1:B": (torch.tensor([1]), torch.tensor([1])),
        "B:r2:C": (torch.tensor([2]), torch.tensor([2])),
        "C:r3:A": (torch.tensor([3]), torch.tensor([3])),
    }
    edges = gb.add_reverse_edges(
        edges, {"A:r1:B": "B:r2:C", "B:r2:C": "C:r3:A", "C:r3:A": "A:r1:B"}
    )
    expected_edges = {
        "A:r1:B": (torch.tensor([1, 3]), torch.tensor([1, 3])),
        "B:r2:C": (torch.tensor([2, 1]), torch.tensor([2, 1])),
        "C:r3:A": (torch.tensor([3, 2]), torch.tensor([3, 2])),
    }
    assert torch.equal(edges["A:r1:B"][0], expected_edges["A:r1:B"][0])
    assert torch.equal(edges["A:r1:B"][1], expected_edges["A:r1:B"][1])
    assert torch.equal(edges["B:r2:C"][0], expected_edges["B:r2:C"][0])
    assert torch.equal(edges["B:r2:C"][1], expected_edges["B:r2:C"][1])
    assert torch.equal(edges["A:r1:B"][0], expected_edges["A:r1:B"][0])
    assert torch.equal(edges["A:r1:B"][1], expected_edges["A:r1:B"][1])
    assert torch.equal(edges["C:r3:A"][0], expected_edges["C:r3:A"][0])
    assert torch.equal(edges["C:r3:A"][1], expected_edges["C:r3:A"][1])


def test_unique_and_compact_hetero():
    N1 = torch.randint(0, 50, (30,))
    N2 = torch.randint(0, 50, (20,))
    N3 = torch.randint(0, 50, (10,))
    unique_N1 = torch.unique(N1)
    unique_N2 = torch.unique(N2)
    unique_N3 = torch.unique(N3)
    expected_unique = {
        "n1": unique_N1,
        "n2": unique_N2,
        "n3": unique_N3,
    }
    nodes_dict = {
        "n1": N1.split(5),
        "n2": N2.split(4),
        "n3": N3.split(2),
    }

    unique, compacted = gb.unique_and_compact(nodes_dict)
    for ntype, nodes in unique.items():
        expected_nodes = expected_unique[ntype]
        assert torch.equal(torch.sort(nodes)[0], expected_nodes)

    for ntype, nodes in compacted.items():
        expected_nodes = nodes_dict[ntype]
        assert isinstance(nodes, list)
        for expected_node, node in zip(expected_nodes, nodes):
            node = unique[ntype][node]
            assert torch.equal(expected_node, node)


def test_unique_and_compact_homo():
    N = torch.randint(0, 50, (200,))
    expected_unique_N = torch.unique(N)
    nodes_list = N.split(5)

    unique, compacted = gb.unique_and_compact(nodes_list)

    assert torch.equal(torch.sort(unique)[0], expected_unique_N)

    assert isinstance(compacted, list)
    for expected_node, node in zip(nodes_list, compacted):
        node = unique[node]
        assert torch.equal(expected_node, node)


def test_unique_and_compact_node_pairs_hetero():
    N1 = torch.randint(0, 50, (30,))
    N2 = torch.randint(0, 50, (20,))
    N3 = torch.randint(0, 50, (10,))
    unique_N1 = torch.unique(N1)
    unique_N2 = torch.unique(N2)
    unique_N3 = torch.unique(N3)
    expected_unique_nodes = {
        "n1": unique_N1,
        "n2": unique_N2,
        "n3": unique_N3,
    }
    node_pairs = {
        "n1:e1:n2": (
            N1[:20],
            N2,
        ),
        "n1:e2:n3": (
            N1[20:30],
            N3,
        ),
        "n2:e3:n3": (
            N2[10:],
            N3,
        ),
    }

    unique_nodes, compacted_node_pairs = gb.unique_and_compact_node_pairs(
        node_pairs
    )
    for ntype, nodes in unique_nodes.items():
        expected_nodes = expected_unique_nodes[ntype]
        assert torch.equal(torch.sort(nodes)[0], expected_nodes)
    for etype, pair in compacted_node_pairs.items():
        u, v = pair
        u_type, _, v_type = gb.etype_str_to_tuple(etype)
        u, v = unique_nodes[u_type][u], unique_nodes[v_type][v]
        expected_u, expected_v = node_pairs[etype]
        assert torch.equal(u, expected_u)
        assert torch.equal(v, expected_v)


def test_unique_and_compact_node_pairs_homo():
    N = torch.randint(0, 50, (200,))
    expected_unique_N = torch.unique(N)

    node_pairs = tuple(N.split(100))
    unique_nodes, compacted_node_pairs = gb.unique_and_compact_node_pairs(
        node_pairs
    )
    assert torch.equal(torch.sort(unique_nodes)[0], expected_unique_N)

    u, v = compacted_node_pairs
    u, v = unique_nodes[u], unique_nodes[v]
    expected_u, expected_v = node_pairs
    unique_v = torch.unique(expected_v)
    assert torch.equal(u, expected_u)
    assert torch.equal(v, expected_v)
    assert torch.equal(unique_nodes[: unique_v.size(0)], unique_v)


def test_incomplete_unique_dst_nodes_():
    node_pairs = (torch.randint(0, 50, (50,)), torch.randint(100, 150, (50,)))
    unique_dst_nodes = torch.arange(150, 200)
    with pytest.raises(IndexError):
        gb.unique_and_compact_node_pairs(node_pairs, unique_dst_nodes)
