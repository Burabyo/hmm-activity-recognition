"""Convert raw Sensor Logger exports into a clean, labelled CSV dataset.

Input  : data/raw/<activity>/<activity>_NN.zip   (training clips, one activity each)
         data/raw/test/test_NN.zip                (unseen continuous recordings)
         data/raw/test/segments.csv               (ground truth: file,activity,start_s,end_s)
Output : data/train/<activity>/<activity>_NN.csv
         data/test/test_NN.csv                    (with a per-sample `label` column;
                                                   unscored moments are labelled `transition`)
         data/metadata.csv                        (per-file sampling-rate report)

Cleaning steps
1. Accelerometer.csv in Sensor Logger is *linear* acceleration (gravity removed);
   we add Gravity.csv back to obtain the true raw accelerometer signal (m/s^2).
2. Streams are aligned on `seconds_elapsed` and resampled onto an exact 100 Hz grid
   by linear interpolation (removes the ~99.3 Hz timing jitter of iOS Core Motion).
3. Training clips are trimmed by TRIM_S seconds at each end to drop the
   start/stop button presses.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
TARGET_HZ = 100.0
TRIM_S = 1.0
ACTIVITIES = ["still", "standing", "walking", "jumping"]


def _read(zf, name):
    with zf.open(name) as f:
        df = pd.read_csv(io.BytesIO(f.read()))
    return df.sort_values("seconds_elapsed").drop_duplicates("seconds_elapsed")


def load_recording(zip_path):
    """Return (resampled DataFrame, measured sampling rate in Hz, raw sample count)."""
    with zipfile.ZipFile(zip_path) as zf:
        lin, grav, gyr = (_read(zf, n) for n in ("Accelerometer.csv", "Gravity.csv", "Gyroscope.csv"))

    t_raw = lin["seconds_elapsed"].to_numpy()
    measured_hz = (len(t_raw) - 1) / (t_raw[-1] - t_raw[0])

    t0 = max(lin.seconds_elapsed.iloc[0], grav.seconds_elapsed.iloc[0], gyr.seconds_elapsed.iloc[0])
    t1 = min(lin.seconds_elapsed.iloc[-1], grav.seconds_elapsed.iloc[-1], gyr.seconds_elapsed.iloc[-1])
    t = np.arange(t0, t1, 1.0 / TARGET_HZ)

    def interp(df, col):
        return np.interp(t, df["seconds_elapsed"], df[col])

    ns0 = lin["time"].iloc[0] - lin["seconds_elapsed"].iloc[0] * 1e9
    out = pd.DataFrame({
        "timestamp_ns": (ns0 + t * 1e9).astype("int64"),
        "seconds_elapsed": np.round(t - t0, 4),
    })
    for ax in "xyz":
        out[f"acc_{ax}"] = interp(lin, ax) + interp(grav, ax)
    for ax in "xyz":
        out[f"gyro_{ax}"] = interp(gyr, ax)
    return out, measured_hz, len(t_raw)


def main():
    rows = []
    for act in ACTIVITIES:
        (OUT / "train" / act).mkdir(parents=True, exist_ok=True)
        for zp in sorted((RAW / act).glob(f"{act}_*.zip")):
            df, hz, n_raw = load_recording(zp)
            dur = df.seconds_elapsed.iloc[-1]
            df = df[(df.seconds_elapsed >= TRIM_S) & (df.seconds_elapsed <= dur - TRIM_S)].copy()
            df["seconds_elapsed"] = np.round(df.seconds_elapsed - TRIM_S, 4)
            df["label"] = act
            df.to_csv(OUT / "train" / act / f"{zp.stem}.csv", index=False)
            rows.append(dict(file=f"train/{act}/{zp.stem}.csv", split="train", activity=act,
                             raw_samples=n_raw, measured_hz=round(hz, 2), raw_duration_s=round(dur, 2),
                             clean_samples=len(df), clean_duration_s=round(len(df) / TARGET_HZ, 2)))

    seg_path = RAW / "test" / "segments.csv"
    if seg_path.exists():
        segs = pd.read_csv(seg_path)
        (OUT / "test").mkdir(parents=True, exist_ok=True)
        for name, s in segs.groupby("file"):
            df, hz, n_raw = load_recording(RAW / "test" / f"{name}.zip")
            df["label"] = "transition"
            for r in s.itertuples():
                df.loc[(df.seconds_elapsed >= r.start_s) & (df.seconds_elapsed < r.end_s), "label"] = r.activity
            # unlabelled moments (e.g. picking the phone up) keep label "transition":
            # the HMM still sees the continuous signal, but they are excluded from scoring
            df.to_csv(OUT / "test" / f"{name}.csv", index=False)
            rows.append(dict(file=f"test/{name}.csv", split="test", activity="mixed",
                             raw_samples=n_raw, measured_hz=round(hz, 2),
                             raw_duration_s=round(df.seconds_elapsed.iloc[-1], 2),
                             clean_samples=len(df), clean_duration_s=round(len(df) / TARGET_HZ, 2)))

    meta = pd.DataFrame(rows)
    meta.to_csv(OUT / "metadata.csv", index=False)
    print(meta.groupby(["split", "activity"]).agg(files=("file", "count"),
                                                  total_s=("clean_duration_s", "sum"),
                                                  mean_hz=("measured_hz", "mean")))


if __name__ == "__main__":
    main()
