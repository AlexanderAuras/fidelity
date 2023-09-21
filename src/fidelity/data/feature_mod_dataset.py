from typing import Any, Iterable, Sequence, Sized, Tuple, Union, cast

import torch.utils.data


class FeatureModificationDataset(torch.utils.data.Dataset[Tuple[Sequence[Any], Sequence[Any]]]):
    def __init__(self, dataset: torch.utils.data.Dataset[Sequence[Any]], input_: Union[int, Iterable[int]] = 0, target: Union[int, Iterable[int]] = 1) -> None:
        super().__init__()
        self.__dataset = dataset
        self.__input_idcs = {input_} if isinstance(input_, int) else set(input_)
        self.__target_idcs = {target} if isinstance(target, int) else set(target)

    def __len__(self) -> int:
        return len(cast(Sized, self.__dataset))

    def __getitem__(self, idx: int) -> Tuple[Sequence[Any], Sequence[Any]]:
        sample = self.__dataset[idx]
        input_ = []
        target = []
        for i, feature in enumerate(sample):
            if i in self.__input_idcs:
                input_.append(feature)
            if i in self.__target_idcs:
                target.append(feature)
        return tuple(input_), tuple(target)
