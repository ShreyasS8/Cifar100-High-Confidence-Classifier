# High-Confidence Image Classification for CIFAR-100

This project addresses the COL774: Machine Learning (Assignment 3) competition, focusing on building a classifier for the CIFAR-100 dataset. The primary goal is not just to achieve high accuracy but to ensure that all predictions are made with very high confidence. The model is designed to abstain from predicting (by outputting a label of `-1`) if it is not sufficiently confident, a critical feature for real-world applications where incorrect predictions have significant consequences.

The model is evaluated using a custom metric that rewards high-accuracy class predictions (≥99%) and heavily penalizes low-accuracy ones.

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

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💾 Dataset Setup

The dataset is provided by the **COL774: Machine Learning (Assignment 3)** Kaggle competition. You must download it manually or use the Kaggle API.

1.  **Download from Kaggle:**
    * Navigate to the competition data page: [https://www.kaggle.com/competitions/col-774-a-3/data](https://www.kaggle.com/competitions/col-774-a-3/data)
    * Download `train.pkl` and `test.pkl`.

2.  **Organize Files:**
    * Create a folder named `data/` in the root of the project.
    * Place the downloaded `.pkl` files inside the `data/` folder.

*(Note: A `.gitignore` file is included to prevent the large data files from being uploaded to the Git repository).*

### Alternative: Using the Kaggle API

For a more automated setup, you can use the Kaggle API.

```bash
# Install Kaggle API
pip install kaggle

# Download and unzip the data into the data/ directory
kaggle competitions download -c col-774-a-3 -p data/
# Note: This competition does not provide a zip file, so you'll get the .pkl files directly.
```

---

## 📖 Usage

### 1. Training the Model

Run the `train.py` script from the command line. You must provide the path to the training data. The `alpha` and `gamma` arguments are included as per the problem description.

```bash
python train.py data/train.pkl 0.99 5
```

This script will train the WideResNet model and save the learned weights to `model.pth`.

### 2. Generating Predictions

After training, run `predict.py` to generate a submission file. This script will load `model.pth`, determine the optimal confidence threshold, and create `predictions.csv`.

```bash
python predict.py
```

This will generate a `predictions.csv` file in the correct format for submission to the Kaggle competition.

---

## 🔬 Methodology

The solution follows a two-stage process to maximize the competition score:

1.  **Model Training**: A WideResNet-34-16 is trained on 90% of the `train.pkl` data. This model learns a robust mapping from images to the 100 CIFAR classes.

2.  **Threshold Tuning & Prediction**:
    * The remaining 10% of the training data is held out as a **validation set**.
    * The trained model predicts on this validation set, and a range of confidence thresholds (e.g., from 0.50 to 0.999) is tested.
    * For each threshold, the official `Final Score` is calculated (`Final Score = High Accuracy Count - γ * Low Accuracy Count`).
    * The threshold that yields the **maximum score** on the validation set is selected as the optimal one.
    * This optimal threshold is then used to make the final predictions on the `test.pkl` data, ensuring the model is perfectly calibrated to the competition's evaluation metric.