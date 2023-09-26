import random
from math import ceil, floor, log2
from typing import Any, Iterable, Optional, Sequence, Sized, Tuple, Union, cast

import torch.utils.data
import xxhash

from fidelity.data.noise import Noise


class FixedNoiseDataset(torch.utils.data.Dataset[Tuple[Sequence[Any], Sequence[Any]]]):
    def __init__(self, dataset: torch.utils.data.Dataset[Tuple[Sequence[Any], Sequence[Any]]], noise: Noise, input_: Union[int, Iterable[int]] = set(), target: Union[int, Iterable[int]] = set(), seed: Optional[int] = None) -> None:  # type: ignore
        super().__init__()
        self.__dataset = dataset
        self.__noise = noise
        self.__input_idcs = {input_} if isinstance(input_, int) else set(input_)
        self.__target_idcs = {target} if isinstance(target, int) else set(target)
        self.__seed = seed if seed is not None else random.randint(0, 999_999_999)

    def __len__(self) -> int:
        return len(cast(Sized, self.__dataset))

    def __getitem__(self, idx: int) -> Tuple[Sequence[Any], Sequence[Any]]:
        sample = self.__dataset[idx]
        input_ = []
        target = []
        for i, feature in enumerate(sample[0]):
            if i not in self.__input_idcs:
                input_.append(feature)
                continue
            hash = xxhash.xxh64_intdigest(idx.to_bytes(floor(log2(idx if idx > 0 else 1)) + 1, "big"), self.__seed) % 2**32
            input_.append(self.__noise(feature, hash))
        for i, feature in enumerate(sample[1]):
            if i not in self.__target_idcs:
                target.append(feature)
                continue
            hash = xxhash.xxh64_intdigest(idx.to_bytes(floor(log2(idx if idx > 0 else 1)) + 1, "big"), self.__seed) % 2**32
            target.append(self.__noise(feature, hash))
        return tuple(input_), tuple(target)
