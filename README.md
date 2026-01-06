# Arrhythmia Classification with Transfer Learning

Real-time Raspberry Pi pipeline that reads PPG from the MAX30102 sensor, computes RR intervals, and scores arrhythmia risk with a transfer-learning CNN. Includes signal-quality checks, a state machine for RR validation, and live Matplotlib visualization.

## Contents
- `main.py`: Sensor stream, bandpass filtering, peak detection, RR state machine, and model inference.
- `ppg_lib/`: MAX30102 driver (`max30102.py`) and package init.
- `cnn_rr_arrhythmia_transfer.h5`: Fine-tuned transfer-learning model.
- `cnn_rr_backbone.weights.h5`: Backbone weights (experimental).
- `data/raw_sessions/`: Sample raw PPG sessions.
- `data/processed/training_features.csv`: RR-based feature set (example training data).
- `ecgg_transfer.ipynb`: Notebook for feature extraction and/or re-training.

## Quickstart
1) Prepare Python 3.9+ (recommended: `python -m venv .venv` then `source .venv/bin/activate` / `.\.venv\Scripts\activate`).
2) Install dependencies: `pip install -r requirements.txt`
3) Wire the MAX30102 via I²C and enable I²C on the Raspberry Pi.
4) Ensure `cnn_rr_arrhythmia_transfer.h5` is present in the project root.
5) Run: `python main.py`  
   - The app shows signal quality and arrhythmia risk on a live plot with colored status text.

## Notes
- If sensor init or I²C access fails, verify the `ppg_lib` path and I²C permissions.
- For long sessions, lower console noise by setting `logging` level to `WARNING`.
- For training/transfer experiments, adapt the steps in `ecgg_transfer.ipynb` to your own datasets.

<p align="right">
  <i>Designed & Developed by <b>Mustafa Gülhan</b></i>
</p>