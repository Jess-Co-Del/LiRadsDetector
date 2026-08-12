#!/usr/bin/env python3
"""AMPLIFAI submission entry point. See amplifai-codabench/SUBMISSION_GUIDE.md.

Expects, alongside this file in the zip:
    lirads_model/           <- our package, including lirads_model/vendor/dinov2_repo
    model/lirads_model.pt   <- trained checkpoint (see lirads_model/train.py --out)
    packages/                <- bundled pip deps (nibabel, ...) built by build.sh
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "packages"))
sys.path.insert(0, _HERE)

import pandas as pd
import torch

from lirads_model import config
from lirads_model.predict import load_model, predict_case

DATA_ROOT = "/app/data/cases"
CHECKPOINT_PATH = os.path.join(_HERE, "model", "lirads_model.pt")


def main() -> None:
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/input_data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/app/output"
    os.makedirs(output_dir, exist_ok=True)

    cases_path = os.path.join(input_dir, "sample_cases.csv")
    cases = pd.read_csv(cases_path)
    case_ids = cases["case_id"].tolist()
    print(f"Processing {len(case_ids)} cases...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, device, backbone_source="local")

    results = []
    for case_id in case_ids:
        case_dir = os.path.join(DATA_ROOT, case_id)
        try:
            prediction = predict_case(model, case_dir, case_id, device)
        except Exception as e:
            print(f"  WARNING: {case_id} failed ({e}); using fallback label", file=sys.stderr)
            prediction = config.FALLBACK_LABEL
        results.append({"case_id": case_id, "prediction": prediction})
        print(f"  {case_id}: {prediction}")

    out_path = os.path.join(output_dir, "predictions.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nDone. {len(results)} predictions written to {out_path}")


if __name__ == "__main__":
    main()
