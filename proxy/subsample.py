"""Cut the three large proxies to 12,000 rows (80/6/14 roles) and park the
full versions in work/proxy_full. Run after build_proxies.py.

    python proxy/subsample.py
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTS = {"afro_r8_multi": "afro_r8_12k", "eph_arg": "eph_p12k", "eph_arg_hh": "eph_hh12k"}


def main():
    os.makedirs(os.path.join(ROOT, "work/proxy_full"), exist_ok=True)
    for src_name, dst_name in CUTS.items():
        src = os.path.join(ROOT, "work/proxy", src_name)
        full = os.path.join(ROOT, "work/proxy_full", src_name)
        if os.path.exists(src):
            shutil.move(src, full)
        schema = json.load(open(os.path.join(full, "schema.json")))
        frame = pd.read_parquet(os.path.join(full, "respondents.parquet"))
        keep = frame.iloc[np.random.default_rng(1).permutation(len(frame))[:12000]].copy()
        keep["role"] = ["TRAIN"] * 9600 + ["DEV"] * 720 + ["TEST"] * 1680
        schema["split"] = {"n_train": 9600, "n_dev": 720, "n_test": 1680}
        schema["dataset"]["n_rows"] = 12000
        dst = os.path.join(ROOT, "work/proxy", dst_name)
        os.makedirs(dst, exist_ok=True)
        keep.to_parquet(os.path.join(dst, "respondents.parquet"), index=False)
        with open(os.path.join(dst, "schema.json"), "w") as fh:
            json.dump(schema, fh)
        print(dst, keep.shape)


if __name__ == "__main__":
    main()
