# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import cached_property

from eval.vsun.reference.reference_dataset import ReferenceDataset
from eval.vsun.reference.reference_dataset_serializer import LMDBGZSerializer


class ReferenceMP2020Correction(ReferenceDataset):
    def __init__(self):
        super().__init__("MP2020correction", ReferenceMP2020Correction.from_preset())

    @classmethod
    def from_preset(cls) -> "ReferenceMP2020Correction":
        return LMDBGZSerializer().deserialize(
            f"eval/vsun/ref_dataset/alex-mp/reference_MP2020correction.gz"
        )

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.

class MP2023(ReferenceDataset):
    def __init__(self):
        super().__init__("mp_02072023", MP2023.from_preset())

    @classmethod
    def from_preset(cls) -> "MP2023":
        return LMDBGZSerializer().deserialize(f"eval/vsun/ref_dataset/mp/mp.gz")

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.
    

class MP20_train(ReferenceDataset):
    def __init__(self):
        super().__init__("mp20_train", MP20_train.from_preset())

    @classmethod
    def from_preset(cls) -> "MP20_train":
        return LMDBGZSerializer().deserialize(f"eval/vsun/ref_dataset/mp20_train/mp20_train.gz")

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.


class Alex_MP_train(ReferenceDataset):
    def __init__(self):
        super().__init__("alex_mp_train", Alex_MP_train.from_preset())

    @classmethod
    def from_preset(cls) -> "Alex_MP_train":
        return LMDBGZSerializer().deserialize(f"eval/vsun/ref_dataset/alex_mp_train/alex_mp_train.gz")

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.
