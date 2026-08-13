from __future__ import annotations

from torch.utils.data import Dataset

from .topcow import TopCoWCase
from .transforms import load_training_arrays, sample_patch


class TopCoWPatchDataset(Dataset):
    def __init__(
        self,
        cases: list[TopCoWCase],
        patch_size: tuple[int, int, int] = (48, 48, 48),
        positive_probability: float = 0.7,
        samples_per_case: int = 4,
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.positive_probability = positive_probability
        self.samples_per_case = samples_per_case

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_case

    def __getitem__(self, index: int):
        case = self.cases[index % len(self.cases)]
        image, label = load_training_arrays(str(case.image), str(case.label))
        image_patch, label_patch = sample_patch(
            image, label, self.patch_size, self.positive_probability
        )
        return {"image": image_patch, "label": label_patch, "case_id": case.case_id}

