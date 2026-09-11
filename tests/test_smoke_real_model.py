"""
Real-model timing smoke test: runs ONE inference pass through the real
(vendored) DINOv2 backbone via predict.predict_case_ensemble -- the same
call submission/run.py makes per case -- and logs the wall-clock time it
took. Meant to be run manually on a machine that actually has the vendored
snapshot (and ideally a GPU), e.g. the HPC node, to get a realistic
per-case inference time estimate ahead of a real 174-case submission run --
NOT as part of the regular fast test suite.

Unlike tests/test_smoke.py (CPU-only, no-internet, uses a tiny random
backbone stub so it runs in seconds), this test needs
lirads_model/vendor/dinov2-with-registers-large to actually be present (see
scripts/vendor_dinov2.sh) and is expected to take real wall-clock time --
DINOv2-large forward passes, times config.TTA_VIEWS+1 views, times however
many checkpoints are given. It's skipped automatically wherever that
vendored snapshot isn't there.

By default it builds an *untrained* LiRadsNet (random head weights) on top
of the real backbone and runs it against one synthetic case, since
inference *time* -- the thing this test measures -- doesn't depend on
whether the head was actually trained; only the (real, expensive) backbone
forward and the TTA/ensemble loop structure do. Point
LIRADS_TEST_CHECKPOINT at one or more real train.py checkpoints
(space-separated) to instead load the real trained model(s) and time the
real ensemble, exactly the way submission/run.py would.

Usage (on the HPC node, from the repo root):
    pytest tests/test_smoke_real_model.py -v -s
    # or, without pytest:
    python tests/test_smoke_real_model.py

    # to time a real (trained) checkpoint / ensemble instead of an untrained head:
    LIRADS_TEST_CHECKPOINT="checkpoints/fold0.pt checkpoints/fold0_seed1.pt" \\
        pytest tests/test_smoke_real_model.py -v -s
"""

import os
import shutil
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lirads_model import config, predict
from lirads_model.backbone import Dinov2SliceEncoder
from lirads_model.model import LiRadsNet


def _make_synthetic_case(case_dir: str, case_id: str, shape=(64, 64, 40)) -> None:
    """Same synthetic-case fixture as tests/test_smoke.py -- see there for
    why it's synthetic rather than real patient data; this test's job is to
    time the real model's inference path, not to validate prediction
    accuracy, so a fabricated but correctly-shaped case exercises the exact
    same preprocessing/backbone/head code path a real one would."""
    ct_dir = os.path.join(case_dir, "ct")
    ann_dir = os.path.join(case_dir, "annotations")
    os.makedirs(ct_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    affine = np.eye(4)

    mask = np.zeros(shape, dtype=np.float32)
    mask[20:36, 20:36, 12:26] = 1.0  # synthetic lesion spanning 14 slices
    nib.save(nib.Nifti1Image(mask, affine), os.path.join(ann_dir, "lesion.nii.gz"))

    for phase in config.PHASE_NAMES:
        vol = rng.normal(50, 100, size=shape).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine), os.path.join(ct_dir, f"{case_id}_{phase}.nii.gz"))


@pytest.mark.skipif(
    not os.path.isdir(config.DINOV2_LOCAL_DIR),
    reason=f"vendored DINOv2 snapshot not found at {config.DINOV2_LOCAL_DIR} -- run scripts/vendor_dinov2.sh first",
)
def test_real_model_inference_time() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths = os.environ.get("LIRADS_TEST_CHECKPOINT", "").split()
    load_start = time.perf_counter()
    if checkpoint_paths:
        models = predict.load_models(checkpoint_paths, device, backbone_source="local")
        print(f"\nLoaded {len(models)} real checkpoint(s): {checkpoint_paths}")
    else:
        backbone = Dinov2SliceEncoder.from_local().to(device)
        model = LiRadsNet(backbone).to(device)
        model.eval()
        models = [model]
        print(
            "\nNo LIRADS_TEST_CHECKPOINT given -- using the real backbone with an untrained "
            "head (fine for timing: inference time doesn't depend on head weights, only the "
            "backbone forward and the TTA/ensemble loop structure do)."
        )
    load_elapsed = time.perf_counter() - load_start

    tmp = tempfile.mkdtemp()
    try:
        case_id = "CASE00001"
        case_dir = os.path.join(tmp, case_id)
        _make_synthetic_case(case_dir, case_id)

        # Same call submission/run.py makes per case (see its main loop),
        # at the real submission defaults (config.TTA_VIEWS,
        # config.MAX_SLICES_PER_CASE) and clinical_features=None -- matching
        # what run.py does whenever no model/clinical/ checkpoints are
        # bundled (see its docstring).
        start = time.perf_counter()
        label = predict.predict_case_ensemble(models, case_dir, case_id, device, clinical_features=None)
        elapsed = time.perf_counter() - start

        msg = (
            f"real-model inference time for 1 case: {elapsed:.2f}s "
            f"(model load: {load_elapsed:.2f}s, {len(models)} checkpoint(s), "
            f"TTA_VIEWS={config.TTA_VIEWS}, MAX_SLICES_PER_CASE={config.MAX_SLICES_PER_CASE}, "
            f"device={device}) -> predicted {label}"
        )
        print(msg)
        config.print_to_log(msg)

        assert label in config.VALID_LABELS
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_real_model_inference_time()
