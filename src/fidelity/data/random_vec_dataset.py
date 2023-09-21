from typing import Any, Sequence, Tuple, Union

import torch.utils.data


class RandomVecDataset(torch.utils.data.Dataset[Tuple[Sequence[Any], Sequence[Any]]]):
    def __init__(self, shape: Union[Sequence[int], int], count: int) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.__vecs = torch.randn((count, *shape))

    def __len__(self) -> int:
        return self.__vecs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Sequence[Any], Sequence[Any]]:
        return (self.__vecs[idx],), tuple()
