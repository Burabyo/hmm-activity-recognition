# Human Activity Recognition with a Hidden Markov Model

**Author:** Jolly Gift Burabyo · Formative 2 – Hidden Markov Models

A Gaussian-emission Hidden Markov Model, implemented **from scratch in NumPy** (Baum–Welch training + Viterbi decoding),
that infers four activities — **still, standing, walking, jumping** — from smartphone accelerometer and gyroscope data.

| | |
|---|---|
| Device | iPhone 11 Pro, Sensor Logger v1.68 |
| Sampling rate | set 100 Hz, measured 99.3 Hz → resampled to exactly 100 Hz |
| Training data | 52 clips (13 per activity, ~10 s each) |
| Unseen test data | 2 continuous multi-activity recordings, different room |
| Test accuracy | **86.2 %** (Viterbi, 80 scored 1-s windows) |

## Repository structure
```
data/
  raw/<activity>/<activity>_NN.zip   original Sensor Logger exports (+ _source_map.txt)
  raw/test/test_NN.zip, segments.csv unseen recordings + ground-truth segment times
  train/<activity>/<activity>_NN.csv cleaned, labelled, 100 Hz training files (52)
  test/test_NN.csv                   cleaned, labelled unseen test files (2)
  metadata.csv                       per-file sampling-rate / duration report
prepare_data.py                      raw exports -> cleaned labelled CSVs
hmm_activity_recognition.ipynb       features, HMM implementation, training, evaluation
figures/                             all figures used in the report
report/HMM_Activity_Report.pdf       final report
videos/video_links.md                data-collection videos (one per activity)
```

CSV columns: `timestamp_ns, seconds_elapsed, acc_x, acc_y, acc_z` (m/s², gravity included), `gyro_x, gyro_y, gyro_z` (rad/s), `label`.

## Data-collection videos
| Activity | Video |
|---|---|
| Standing | https://drive.google.com/file/d/1yA_DP3qB3pFlJeSCUPvisG_z1yE_k0aB/view?usp=sharing |
| Walking  | https://drive.google.com/file/d/1Vyu1zIJ2cXOFrPhvDf8uZc9PPhVjlmmy/view?usp=sharing |
| Jumping  | https://drive.google.com/file/d/1cam80EsMgi6tWBkMEZ8WwPjal0GsLBZj/view?usp=sharing |
| Still    | https://drive.google.com/file/d/12Muhzl1npKDIN6ipxp3qi3YnDbDqRG-c/view?usp=sharing |

## Reproduce
```bash
pip install -r requirements.txt
python prepare_data.py              # rebuilds data/train, data/test, data/metadata.csv
jupyter notebook hmm_activity_recognition.ipynb
```
