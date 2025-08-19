# High-Confidence Image Classification for CIFAR-100

This project focuses on building a classifier for the CIFAR-100 dataset. The primary goal is not just to achieve high accuracy but to ensure that all predictions are made with very high confidence. The model is designed to abstain from predicting (by outputting a label of `-1`) if it is not sufficiently confident — a critical feature for real-world applications where incorrect predictions have significant consequences.

---

## 🚀 Features

* **Model**: Implements a **Wide Residual Network (WideResNet-34-16)** from scratch using PyTorch for robust feature extraction.
* **Data Augmentation**: Utilizes random cropping and horizontal flipping during training to improve model generalization.
* **Confidence-Based Prediction**: Employs a confidence thresholding mechanism to decide whether to classify an image or abstain.
* **Custom Metric Optimization**: The confidence threshold is strategically tuned on a validation set to maximize the competition's specific evaluation score.

---

## 📂 Project Structure

```
.
├── data/
│   ├── train.pkl     # Training data (downloaded from Kaggle)
│   └── test.pkl      # Test data (downloaded from Kaggle)
├── .gitignore        # Tells Git to ignore the data/ folder
├── train.py          # Script to train the model
├── predict.py        # Script to generate predictions for submission
├── requirements.txt  # Python dependencies
├── model.pth         # Saved model weights (generated after training)
└── README.md         # This file
```

---

## 🔧 Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 💾 Dataset Setup

The dataset is provided by the **COL774: Machine Learning (Assignment 3)** Kaggle competition. You must download it manually or use the Kaggle API.

1. **Download from Kaggle:**

   * Navigate to the competition data page: `https://www.kaggle.com/competitions/col-774-a-3/data`
   * Download `train.pkl` and `test.pkl`.

2. **Organize Files:**

   * Create a folder named `data/` in the root of the project.
   * Place the downloaded `.pkl` files inside the `data/` folder.

*(Note: A `.gitignore` file is included to prevent the large data files from being uploaded to the Git repository).*

### Alternative: Using the Kaggle API

For a more automated setup, you can use the Kaggle API.

```bash
# Install Kaggle API
pip install kaggle

# Download the data into the data/ directory
kaggle competitions download -c col-774-a-3 -p data/
# Note: This competition provides the .pkl files directly.
```

---

## 📖 Usage

> **Important:** The `predict.py` in this repository requires four command-line arguments: the model path, the test `.pkl` path, `alpha` (confidence threshold), and `gamma` (competition penalty parameter). Example and details below.

### 1. Training the Model

Run the `train.py` script from the command line. You must provide the path to the training data and (optionally) the `alpha` and `gamma` parameters if your `train.py` accepts those as shown in the example below.

```bash
python train.py data/train.pkl 0.99 5
```

This script will train the WideResNet model and save the learned weights (for example `model.pth`). **(Leave `train.py` unchanged unless you want to modify training hyperparameters or checkpointing behaviour.)**

### 2. Generating Predictions (Updated)

The `predict.py` script now expects the following command-line arguments:

```bash
python predict.py <path_to_model.pth> <path_to_test.pkl> <alpha> <gamma>
```

* `<path_to_model.pth>` — the path to the saved model weights (PyTorch `state_dict`). Example: `model.pth`.
* `<path_to_test.pkl>` — path to the test set file (the same format used for training/validation in the repo). Example: `data/test.pkl`.
* `<alpha>` — confidence threshold in `[0, 1]`. Predictions with top softmax probability below `alpha` will be set to `-1` (abstain). Example: `0.99`.
* `<gamma>` — the competition penalty parameter (kept for compatibility with the repository and for downstream scoring). **NOTE:** The current `predict.py` implementation reads `gamma` from `sys.argv` but does not use it in the prediction loop — it is retained so the same CLI can be used across `train.py` and `predict.py`.

**Example:**

```bash
python predict.py model.pth data/test.pkl 0.99 5
```

**What the script does:**

* Loads the WideResNet-34-16 model and the provided `model.pth` weights.
* Iterates over the test dataset and computes softmax probabilities for each sample.
* For each sample, if the maximum softmax probability is >= `alpha`, it outputs the predicted label; otherwise it outputs `-1`.
* The script uses a built-in temperature scaling list (the current code uses `temperatures = [2.95]`). If you want to test different temperatures, edit `predict.py` and change the `temperatures` list or add a command-line parameter.
* The output file is written as `submission.csv` (in the repository root) by default. Each row has columns `ID` and `Predicted_label`.

**Output file format (example):**

```
ID,Predicted_label
0,-1
1,12
2,56
...
```

---

## 🔬 Methodology

The solution follows a two-stage process to maximize the competition score:

1. **Model Training**: A WideResNet-34-16 is trained on 90% of the `train.pkl` data. This model learns a robust mapping from images to the 100 CIFAR classes.

2. **Threshold Tuning & Prediction**:

   * The remaining 10% of the training data is held out as a **validation set**.
   * The trained model predicts on this validation set, and a range of confidence thresholds (e.g., from 0.50 to 0.999) is tested.
   * For each threshold, the official `Final Score` is calculated (`Final Score = High Accuracy Count - γ * Low Accuracy Count`).
   * The threshold that yields the **maximum score** on the validation set is selected as the optimal one.
   * This optimal threshold is then used to make the final predictions on the `test.pkl` data, ensuring the model is calibrated to the competition's evaluation metric.

---

## 📝 Notes & Troubleshooting

* **Model format:** Save and load the weights with `torch.save(model.state_dict(), 'model.pth')` and `model.load_state_dict(torch.load('model.pth'))` as used in `predict.py`.
* **GPU usage:** The scripts automatically pick `cuda` when available. If you want to force CPU, either set `CUDA_VISIBLE_DEVICES=` in your environment or modify the device selection in the scripts.
* **Confidence threshold (`alpha`) selection:** `alpha` should be tuned on a held-out validation set to maximize your custom metric. The `train.py` should perform this search and record the best value (if implemented).
* **Temperature scaling:** The current `predict.py` contains a hard-coded `temperatures` list (default `[2.95]`). If you want temperature to be an argument, add a CLI argument and pass it through into the prediction loop.
* **Gamma parameter:** The `gamma` CLI argument is provided for compatibility with the competition scoring but is currently unused in `predict.py` — include/implement it if you want post-hoc scoring in the prediction script.

---

If you want, I can also:

* Add an optional CLI argument for temperature(s).
* Make `predict.py` write a configurable output filename (instead of `submission.csv`).
* Add a quick script to compute the competition `Final Score` from a labeled validation set and a `predictions.csv`.

Tell me which of those (if any) you want next.
