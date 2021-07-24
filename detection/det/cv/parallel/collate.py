from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from .data_container import DataContainer


def collate(batch, samples_per_gpu=1):
    """
    Extend `default_collate` to add support for `DataContainer`. 3 cases:

    1. cpu_only = True, e.g. meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., groud truth boxes
    """
    elem = batch[0]
    if isinstance(elem, DataContainer):
        stacked = []
        if elem.cpu_only:
            # leave the objects as is
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i : i + samples_per_gpu]]
                )
            return DataContainer(stacked, elem.stack, elem.padding_value, cpu_only=True)

        elif elem.stack:
            for i in range(0, len(batch), samples_per_gpu):
                elem_i = batch[i]
                assert isinstance(elem_i.data, torch.Tensor)

                if elem_i.pad_dims is not None:
                    ndim = elem_i.dim()
                    assert ndim > elem_i.pad_dims

                    max_shape = [0] * elem_i.pad_dims
                    # fill `max_shape` by iterating through all `samples_per_gpu` samples
                    for sample in batch[i : i + samples_per_gpu]:
                        # should be of the same size except for the last `pad_dims` dimensions
                        assert all(
                            elem_i.size(dim) == sample.size(dim)
                            for dim in range(ndim - elem.pad_dims)
                        )
                        max_shape = [
                            max(max_shape[dim - 1], sample.size(-dim))
                            for dim in range(1, elem_i.pad_dims + 1)
                        ]

                    padded_samples = []
                    for sample in batch[i : i + samples_per_gpu]:
                        pad = [0] * (elem_i.pad_dims * 2)
                        for dim in range(1, elem_i.pad_dims + 1):
                            # `pad` is nonzero only on odd indices.
                            pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(sample.data, pad, value=sample.padding_value)
                        )

                    stacked.append(default_collate(padded_samples))
                elif elem_i.pad_dims is None:
                    # stacking without padding
                    stacked.append(
                        default_collate(
                            sample.data for sample in batch[i : i + samples_per_gpu]
                        )
                    )
                else:
                    raise ValueError
        else:
            # no stacking
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i : i + samples_per_gpu]]
                )
        return DataContainer(stacked, elem.stack, elem.padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu) for key in batch[0]
        }
    else:
        return default_collate(batch)
