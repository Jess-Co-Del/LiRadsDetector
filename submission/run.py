#!/usr/bin/env python3
"""

AMPLIFAI Challenge — submission entry point.

Expects, alongside this file in the zip:
    lirads_model/           <- our package, including
                                lirads_model/vendor/dinov2-with-registers-large
    model/*.pt               <- trained checkpoint(s) (see lirads_model/train.py --out).
                                One file predicts normally; more than one is
                                majority-voted across models per case.
    model/clinical/*.pt      <- optional trained clinical-predictor checkpoint(s)
                                (see lirads_model/train_clinical.py --out). When
                                present, predict_case_ensemble synthesizes a
                                clinical/tabular feature row per case
                                (lirads_model/predict_clinical.py) and feeds it to
                                the main model(s)' clinical branch, since the real
                                challenge input never supplies one itself --
                                sharing the same deterministic backbone pass its
                                own main-ensemble prediction uses, rather than
                                running the clinical stage as a separate pass over
                                all cases first (see predict.predict_case_ensemble's
                                docstring). Absent this directory, the main
                                model(s) fall back to their learned
                                missing-clinical embedding.
    packages/                <- bundled pip deps (nibabel, ...) built by build.sh
"""

import glob
import os
import sys
from datetime import datetime
from time import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "packages"))
sys.path.insert(0, _HERE)
import transformers
print('TRANSFORMERS VERSION', transformers.__version__)
import pandas as pd
import torch

from lirads_model import config
from lirads_model.predict import load_models, predict_case_ensemble
from lirads_model.predict_clinical import load_clinical_models

DATA_ROOT = "/leonardo_scratch/fast/EUHPC_D35_139/nnunet_base/nnunet_format/amplifai/batch_001/cases"  # "/app/data/cases"
MODEL_DIR = os.path.join(_HERE, "model")
CLINICAL_MODEL_DIR = os.path.join(MODEL_DIR, "clinical")


def _log(msg: str) -> None:
    print(f"{datetime.fromtimestamp(time())}: {msg}")


def main() -> None:
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/input_data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/app/output"
    os.makedirs(output_dir, exist_ok=True)

    cases_path = os.path.join(input_dir, "sample_cases.csv")
    cases = pd.read_csv(cases_path)
    case_ids = cases["case_id"].tolist()
    _log(f"Processing {len(case_ids)} cases...")

    checkpoint_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))
    if not checkpoint_paths:
        raise FileNotFoundError(f"no .pt checkpoints found in {MODEL_DIR}")
    print(f"Loading {len(checkpoint_paths)} model(s): {[os.path.basename(p) for p in checkpoint_paths]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(checkpoint_paths, device, backbone_source="local")

    clinical_models = None
    clinical_checkpoint_paths = sorted(glob.glob(os.path.join(CLINICAL_MODEL_DIR, "*.pt")))
    if clinical_checkpoint_paths:
        print(f"Loading {len(clinical_checkpoint_paths)} clinical-predictor model(s): {[os.path.basename(p) for p in clinical_checkpoint_paths]}")
        clinical_models = load_clinical_models(clinical_checkpoint_paths, device, backbone_source="local")
    else:
        print(f"  no clinical-predictor checkpoints found in {CLINICAL_MODEL_DIR}; "
              "main model(s) will use their learned missing-clinical embedding for every case")

    results = []
    clinical_rows = []
    for case_id in case_ids:
        case_dir = os.path.join(DATA_ROOT, case_id)
        try:
            prediction, clinical_row = predict_case_ensemble(models, case_dir, case_id, device, clinical_models=clinical_models)
        except Exception as e:
            print(f"  WARNING: {case_id} failed ({e}); using fallback label", file=sys.stderr)
            prediction, clinical_row = config.FALLBACK_LABEL, None
        if clinical_row is not None:
            clinical_rows.append(clinical_row)
        results.append({"case_id": case_id, "prediction": prediction})
        _log(f"{case_id}: {prediction}")

    if clinical_rows:
        metadata_out_path = os.path.join(output_dir, "generated_metadata.csv")
        pd.DataFrame(clinical_rows).to_csv(metadata_out_path, index=False)
        print(f"wrote generated clinical metadata for {len(clinical_rows)}/{len(case_ids)} case(s) to {metadata_out_path}")

    out_path = os.path.join(output_dir, "predictions.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    _log(f"Done. {len(results)} predictions written to {out_path}")


if __name__ == "__main__":
    main()
