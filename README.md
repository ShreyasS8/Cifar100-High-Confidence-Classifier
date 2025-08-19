# High-Confidence Image Classification for CIFAR-100

This repository implements a high-confidence classifier for the CIFAR-100 dataset. Its goal is not only to get correct labels, but to **only predict when it is very confident** — otherwise it abstains by outputting `-1`. This is intended for settings where incorrect predictions are costly and abstaining is preferable.

---

## 🚀 Features

- **Model**: WideResNet-34-16 implemented in PyTorch.
- **Abstention / Confidence Thresholding**: Predictions with maximum softmax probability below an `alpha` threshold are set to `-1`.
- **Temperature scaling**: Optionally scale logits by a temperature before softmax.
- **Validation-based threshold tuning**: Intended workflow tunes `alpha` on a validation split to maximize a custom competition metric.
- **Simple scripts**: `train.py` (training) and `predict.py` (inference / submit generation).

---

## 📂 Project structure

```
.
├── LICENSE
├── README.md
├── predict.py       # Inference / submission generation (expects CLI args)
├── requirements.txt
└── train.py         # Training script
```

(Training / test data should be placed in a `data/` folder — see Dataset Setup below.)

---

## 🔧 Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. (Recommended) Create & activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate         # macOS / Linux
# venv\Scripts\activate          # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, a minimum set is:
```
torch
torchvision
numpy
pandas
tqdm
Pillow
opencv-python
matplotlib
```

---

## 💾 Dataset Setup

This project uses the CIFAR-100 formatted `.pkl` files from the COL774 Kaggle assignment.

1. Download `train.pkl` and `test.pkl` from:
   `https://www.kaggle.com/competitions/col-774-a-3/data`

2. Place the files into a folder `data/` at the repo root:
```
data/train.pkl
data/test.pkl
```

Alternative (Kaggle API):
```bash
pip install kaggle
kaggle competitions download -c col-774-a-3 -p data/
```

---

## 📖 Usage

> **Important:** `predict.py` in this repo expects **four** positional CLI arguments:
> 1. path to the saved model weights (`.pth`)  
> 2. path to the test `.pkl` file  
> 3. `alpha` — confidence threshold (float between 0 and 1)  
> 4. `gamma` — competition penalty parameter (float) — included for CLI compatibility / downstream scoring

### 1) Training
Run the training script. Example (if `train.py` accepts `alpha`/`gamma` as shown in the repository):
```bash
python train.py data/train.pkl 0.99 5
```
This will train the WideResNet and (expectedly) save a `model.pth` (or similar checkpoint). See `train.py` for details about checkpoint naming and training options.

### 2) Generating predictions (inference)
Run:
```bash
python predict.py <path_to_model.pth> <path_to_test.pkl> <alpha> <gamma>
```

Example:
```bash
python predict.py model.pth data/test.pkl 0.99 5
```

**What `predict.py` does (current implementation):**
- Loads WideResNet-34-16 and `model.load_state_dict(torch.load(...))`.
- Builds a test `DataLoader` from the provided `.pkl`.
- For each test batch:
  - Computes logits, optionally scales them by a temperature value from a list (the current code uses `temperatures = [2.95]`).
  - Applies softmax to get class probabilities.
  - Computes the maximum softmax probability (confidence) and the corresponding predicted label.
  - If `confidence >= alpha` → keep predicted label; else → `-1`.
- Writes output to `submission.csv` in the repo root with columns: `ID,Predicted_label`.

**Output format (submission.csv)**
```
ID,Predicted_label
0,-1
1,12
2,56
...
```

**Notes about CLI args**
- `alpha` (confidence threshold) is actively used to decide abstain vs predict.
- `gamma` is accepted by the script for compatibility with scoring logic, but the current `predict.py` does not use `gamma` inside the inference loop — it's intended for evaluation / scoring steps (or future extensions).
- Temperature scaling is currently hard-coded. If you want temperature to be a CLI argument, you can modify `predict.py` to read an additional argument or add an `argparse` flag.

---

## 🔬 Methodology (brief)

1. **Train** a WideResNet-34-16 on the training split (e.g., 90% train / 10% validation).
2. **Tune `alpha`** on the held-out validation set. For each candidate `alpha`, compute the competition metric:
   - `Final Score = HighAccuracyCount - γ * LowAccuracyCount`
   where `HighAccuracyCount` is the number of classes with per-class accuracy ≥ 99% (or per the competition definition) — adapt this computation to match the exact evaluation rules.
3. Use the `alpha` that maximizes the validation score to generate predictions on `test.pkl`.

---

## 🛠️ Recommendations & Improvements (optional)

If you'd like to improve the repo or make it more flexible, consider implementing:
- CLI parsing using `argparse` (for optional temperature, output path, batch size, device).
- Add an `--output` argument to `predict.py` to write a custom filename instead of `submission.csv`.
- Allow `predict.py` to accept a list of temperatures and compute/compare results, or read the best temperature from a saved config file.
- Implement a `score.py` (or extend `predict.py`) that computes the competition `Final Score` on a labeled validation set using the same thresholding logic.
- Persist the selected best `alpha` (and temperature) in `train.py` so you can reuse them for inference automatically.

---

## 📝 Troubleshooting

- **Model load error**: Ensure `model.pth` was saved with `model.state_dict()` and you load it via `model.load_state_dict(torch.load(...))`.
- **CUDA/CPU**: The scripts auto-select CUDA if available. To force CPU, set `CUDA_VISIBLE_DEVICES=` or edit the device selection.
- **Mismatch in `.pkl` format**: `predict.py` expects the test `.pkl` to contain a list-like structure where each element is `(image, label)` or similar — inspect the loader in the script if your `.pkl` differs.
- **Temperature / alpha tuning**: For best performance, run a search over `alpha` (e.g., `np.linspace(0.5, 0.999, 100)`) on the validation set and choose the `alpha` that maximizes the custom score.

---

## Contributing

Contributions are welcome. Typical improvements:
- Add CLI / argparse for `train.py` and `predict.py`.
- Add unit tests for dataset loader and scoring function.
- Add a `Makefile` or Dockerfile for reproducible runs.

---

## License

See the `LICENSE` file in the repository.

---

If you want, I can:
- Edit `predict.py` to accept optional flags: `--temperature`, `--output`, `--batch-size`.
- Add a `score.py` that calculates your competition `Final Score` on a labeled validation set.
- Add `argparse` to both `train.py` and `predict.py` and produce usage examples.

Which of those would you like me to implement next?
