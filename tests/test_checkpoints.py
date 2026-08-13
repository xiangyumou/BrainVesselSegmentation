from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bvs.checkpoints import (
    load_lingfeng_student_checkpoint,
    verify_lingfeng_equivalence,
)
from bvs.models import LingfengMRAStudent

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/checkpoints/lingfeng/student_best_checkpoint_multimodaltune9.pt"
)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Lingfeng checkpoint is not installed")
def test_checkpoint_loads_all_required_student_keys() -> None:
    report = load_lingfeng_student_checkpoint(LingfengMRAStudent(), CHECKPOINT)
    assert len(report["loaded_keys"]) == 25
    assert report["ignored_keys"]
    assert report["sha256"] == "ccecc4b52ffa3832ebf2580945b19e71315f2c26c7f0149f6ecd099ca0997a22"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Lingfeng checkpoint is not installed")
def test_legacy_student_logits_are_equivalent() -> None:
    report = verify_lingfeng_equivalence(CHECKPOINT, "cpu", patch_size=16)
    assert report["equivalent"]
    assert report["max_abs_error"] <= 1e-5


def test_missing_student_key_fails_loudly(tmp_path: Path) -> None:
    model = LingfengMRAStudent()
    state = model.state_dict()
    state.pop(next(iter(state)))
    checkpoint = tmp_path / "incomplete.pt"
    torch.save({"model": state}, checkpoint)
    with pytest.raises(RuntimeError, match="missing="):
        load_lingfeng_student_checkpoint(model, checkpoint)
