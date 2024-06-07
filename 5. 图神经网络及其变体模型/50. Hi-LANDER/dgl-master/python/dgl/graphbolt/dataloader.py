"""Graph Bolt DataLoaders"""

import torch.utils.data
import torchdata.dataloader2.graph as dp_utils
import torchdata.datapipes as dp

from .base import CopyTo
from .feature_fetcher import FeatureFetcher
from .item_sampler import ItemSampler

from .utils import datapipe_graph_to_adjlist


__all__ = [
    "SingleProcessDataLoader",
    "MultiProcessDataLoader",
]


def _find_and_wrap_parent(
    datapipe_graph, datapipe_adjlist, target_datapipe, wrapper, **kwargs
):
    """Find parent of target_datapipe and wrap it with ."""
    datapipes = dp_utils.find_dps(
        datapipe_graph,
        target_datapipe,
    )
    for datapipe in datapipes:
        datapipe_id = id(datapipe)
        for parent_datapipe_id in datapipe_adjlist[datapipe_id][1]:
            parent_datapipe, _ = datapipe_adjlist[parent_datapipe_id]
            datapipe_graph = dp_utils.replace_dp(
                datapipe_graph,
                parent_datapipe,
                wrapper(parent_datapipe, **kwargs),
            )


class SingleProcessDataLoader(torch.utils.data.DataLoader):
    """Single process DataLoader.

    Iterates over the data pipeline in the main process.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    """

    # In the single process dataloader case, we don't need to do any
    # modifications to the datapipe, and we just PyTorch's native
    # dataloader as-is.
    #
    # The exception is that batch_size should be None, since we already
    # have minibatch sampling and collating in ItemSampler.
    def __init__(self, datapipe):
        datapipe_graph = dp_utils.traverse_dps(datapipe)
        datapipe_adjlist = datapipe_graph_to_adjlist(datapipe_graph)
        # Cut datapipe at CopyTo and wrap with prefetcher. This enables the
        # data pipeline up to the CopyTo operation to run in a separate thread.
        _find_and_wrap_parent(
            datapipe_graph,
            datapipe_adjlist,
            CopyTo,
            dp.iter.Prefetcher,
            buffer_size=2,
        )
        super().__init__(datapipe, batch_size=None, num_workers=0)


class MultiprocessingWrapper(dp.iter.IterDataPipe):
    """Wraps a datapipe with multiprocessing.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    num_workers : int, optional
        The number of worker processes. Default is 0, meaning that there
        will be no multiprocessing.
    """

    def __init__(self, datapipe, num_workers=0):
        self.datapipe = datapipe
        self.dataloader = torch.utils.data.DataLoader(
            datapipe,
            batch_size=None,
            num_workers=num_workers,
        )

    def __iter__(self):
        yield from self.dataloader


class MultiProcessDataLoader(torch.utils.data.DataLoader):
    """Multiprocessing DataLoader.

    Iterates over the data pipeline with everything before feature fetching
    (i.e. :class:`dgl.graphbolt.FeatureFetcher`) in subprocesses, and
    everything after feature fetching in the main process. The datapipe
    is modified in-place as a result.

    Only works on single GPU.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    num_workers : int, optional
        Number of worker processes. Default is 0, which is identical to
        :class:`SingleProcessDataLoader`.
    """

    def __init__(self, datapipe, num_workers=0):
        # Multiprocessing requires two modifications to the datapipe:
        #
        # 1. Insert a stage after ItemSampler to distribute the
        #    minibatches evenly across processes.
        # 2. Cut the datapipe at FeatureFetcher, and wrap the inner datapipe
        #    of the FeatureFetcher with a multiprocessing PyTorch DataLoader.

        datapipe_graph = dp_utils.traverse_dps(datapipe)
        datapipe_adjlist = datapipe_graph_to_adjlist(datapipe_graph)

        # (1) Insert minibatch distribution.
        # TODO(BarclayII): Currently I'm using sharding_filter() as a
        # concept demonstration. Later on minibatch distribution should be
        # merged into ItemSampler to maximize efficiency.
        item_samplers = dp_utils.find_dps(
            datapipe_graph,
            ItemSampler,
        )
        for item_sampler in item_samplers:
            datapipe_graph = dp_utils.replace_dp(
                datapipe_graph,
                item_sampler,
                item_sampler.sharding_filter(),
            )

        # (2) Cut datapipe at FeatureFetcher and wrap.
        _find_and_wrap_parent(
            datapipe_graph,
            datapipe_adjlist,
            FeatureFetcher,
            MultiprocessingWrapper,
            num_workers=num_workers,
        )

        # (3) Cut datapipe at CopyTo and wrap with prefetcher. This enables the
        # data pipeline up to the CopyTo operation to run in a separate thread.
        _find_and_wrap_parent(
            datapipe_graph,
            datapipe_adjlist,
            CopyTo,
            dp.iter.Prefetcher,
            buffer_size=2,
        )

        # The stages after feature fetching is still done in the main process.
        # So we set num_workers to 0 here.
        super().__init__(datapipe, batch_size=None, num_workers=0)
