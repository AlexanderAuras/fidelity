from typing import Any, Callable, Mapping, Sequence, Sized, Tuple, cast

import torch.utils.data


class TransformDataset(torch.utils.data.Dataset[Tuple[Sequence[Any], Sequence[Any]]]):
    def __init__(self, dataset: torch.utils.data.Dataset[Tuple[Sequence[Any], Sequence[Any]]], input_transforms: Mapping[int, Callable[[Any], Any]] = {}, target_transforms: Mapping[int, Callable[[Any], Any]] = {}) -> None:  # type: ignore
        super().__init__()
        self.__dataset = dataset
        self.__input_transforms = input_transforms
        self.__target_transforms = target_transforms

    def __len__(self) -> int:
        return len(cast(Sized, self.__dataset))

    def __getitem__(self, idx: int) -> Tuple[Sequence[Any], Sequence[Any]]:
        sample = self.__dataset[idx]
        input_ = []
        target = []
        for i, feature in enumerate(sample[0]):
            input_.append(self.__input_transforms[i](feature) if i in self.__input_transforms else feature)
        for i, feature in enumerate(sample[1]):
            target.append(self.__target_transforms[i](feature) if i in self.__target_transforms else feature)
        return tuple(input_), tuple(target)
