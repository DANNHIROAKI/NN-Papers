import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch
from torchdata.datapipes.iter import Mapper


def test_SubgraphSampler_invoke():
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)

    # Invoke via class constructor.
    datapipe = gb.SubgraphSampler(item_sampler)
    with pytest.raises(NotImplementedError):
        next(iter(datapipe))

    # Invokde via functional form.
    datapipe = item_sampler.sample_subgraph()
    with pytest.raises(NotImplementedError):
        next(iter(datapipe))


@pytest.mark.parametrize("labor", [False, True])
def test_NeighborSampler_invoke(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    # Invoke via class constructor.
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(item_sampler, graph, fanouts)
    assert len(list(datapipe)) == 5

    # Invokde via functional form.
    if labor:
        datapipe = item_sampler.sample_layer_neighbor(graph, fanouts)
    else:
        datapipe = item_sampler.sample_neighbor(graph, fanouts)
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_NeighborSampler_fanouts(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2

    # `fanouts` is a list of tensors.
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    if labor:
        datapipe = item_sampler.sample_layer_neighbor(graph, fanouts)
    else:
        datapipe = item_sampler.sample_neighbor(graph, fanouts)
    assert len(list(datapipe)) == 5

    # `fanouts` is a list of integers.
    fanouts = [2 for _ in range(num_layer)]
    if labor:
        datapipe = item_sampler.sample_layer_neighbor(graph, fanouts)
    else:
        datapipe = item_sampler.sample_neighbor(graph, fanouts)
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Node(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    sampler_dp = Sampler(item_sampler, graph, fanouts)
    assert len(list(sampler_dp)) == 5


def to_link_batch(data):
    block = gb.MiniBatch(node_pairs=data)
    return block


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2), names="node_pairs")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(item_sampler, graph, fanouts)
    assert len(list(neighbor_dp)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link_With_Negative(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2), names="node_pairs")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    negative_dp = gb.UniformNegativeSampler(item_sampler, graph, 1)
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(negative_dp, graph, fanouts)
    assert len(list(neighbor_dp)) == 5


def get_hetero_graph():
    # COO graph:
    # [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # [2, 4, 2, 3, 0, 1, 1, 0, 0, 1]
    # [1, 1, 1, 1, 0, 0, 0, 0, 0] - > edge type.
    # num_nodes = 5, num_n1 = 2, num_n2 = 3
    ntypes = {"n1": 0, "n2": 1}
    etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
    metadata = gb.GraphMetadata(ntypes, etypes)
    indptr = torch.LongTensor([0, 2, 4, 6, 8, 10])
    indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 0, 1])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    node_type_offset = torch.LongTensor([0, 2, 5])
    return gb.from_fused_csc(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        metadata=metadata,
    )


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Node_Hetero(labor):
    graph = get_hetero_graph()
    itemset = gb.ItemSetDict(
        {"n2": gb.ItemSet(torch.arange(3), names="seed_nodes")}
    )
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    sampler_dp = Sampler(item_sampler, graph, fanouts)
    assert len(list(sampler_dp)) == 2
    for minibatch in sampler_dp:
        assert len(minibatch.sampled_subgraphs) == num_layer


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link_Hetero(labor):
    graph = get_hetero_graph()
    itemset = gb.ItemSetDict(
        {
            "n1:e1:n2": gb.ItemSet(
                torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T,
                names="node_pairs",
            ),
            "n2:e2:n1": gb.ItemSet(
                torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T,
                names="node_pairs",
            ),
        }
    )

    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(item_sampler, graph, fanouts)
    assert len(list(neighbor_dp)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link_Hetero_With_Negative(labor):
    graph = get_hetero_graph()
    itemset = gb.ItemSetDict(
        {
            "n1:e1:n2": gb.ItemSet(
                torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T,
                names="node_pairs",
            ),
            "n2:e2:n1": gb.ItemSet(
                torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T,
                names="node_pairs",
            ),
        }
    )

    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    negative_dp = gb.UniformNegativeSampler(item_sampler, graph, 1)
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(negative_dp, graph, fanouts)
    assert len(list(neighbor_dp)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Random_Hetero_Graph(labor):
    num_nodes = 5
    num_edges = 9
    num_ntypes = 3
    num_etypes = 3
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        metadata,
    ) = gb_test_utils.random_hetero_graph(
        num_nodes, num_edges, num_ntypes, num_etypes
    )
    edge_attributes = {
        "A1": torch.randn(num_edges),
        "A2": torch.randn(num_edges),
    }
    graph = gb.from_fused_csc(
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        edge_attributes,
        metadata,
    )
    itemset = gb.ItemSetDict(
        {
            "n2": gb.ItemSet(torch.tensor([0]), names="seed_nodes"),
            "n1": gb.ItemSet(torch.tensor([1]), names="seed_nodes"),
        }
    )

    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    sampler_dp = Sampler(item_sampler, graph, fanouts, replace=True)

    for data in sampler_dp:
        for sampledsubgraph in data.sampled_subgraphs:
            for _, value in sampledsubgraph.node_pairs.items():
                assert torch.equal(
                    torch.ge(value[0], torch.zeros(len(value[0]))),
                    torch.ones(len(value[0])),
                )
                assert torch.equal(
                    torch.ge(value[1], torch.zeros(len(value[1]))),
                    torch.ones(len(value[1])),
                )
            for _, value in sampledsubgraph.original_column_node_ids.items():
                assert torch.equal(
                    torch.ge(value, torch.zeros(len(value))),
                    torch.ones(len(value)),
                )
            for _, value in sampledsubgraph.original_row_node_ids.items():
                assert torch.equal(
                    torch.ge(value, torch.zeros(len(value))),
                    torch.ones(len(value)),
                )
