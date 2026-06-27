# 🏀 Predicting NBA Draft Outcomes from NCAA Player Statistics

> **Cornell Tech — PAML Final Project**
>
> Can we predict whether a college basketball player will be drafted into the NBA — and if so, in which round — using only their NCAA statistics?

This project builds and compares three classification models **from scratch** (NumPy / Pandas only, no scikit-learn / PyTorch / TensorFlow) to predict a player's NBA draft outcome as one of three classes:

| Class | Label          |
| :---: | -------------- |
|   0   | Undrafted      |
|   1   | 1st Round Pick |
|   2   | 2nd Round Pick |

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Getting Started](#getting-started)
- [Reports](#reports)

---

## Dataset

We use the **Kaggle College Basketball 2009–2021 + NBA Advanced Stats** dataset, which contains per-season statistics for NCAA Division I players along with their NBA draft outcomes.

| Split      |   Seasons   |   Rows |
| :--------- | :---------: | -----: |
| Train      | 2009 – 2018 | 16,846 |
| Validation | 2019 – 2020 |  3,643 |
| Test       |    2021     |  4,956 |

**Features:** 63 numeric columns (points, rebounds, assists, shooting percentages, advanced stats such as BPM, usage rate, recruiting rank, height, etc.) plus 3 categorical columns (team, conference, role/position).

**Key challenge:** Severe class imbalance — over **96 %** of players go undrafted, making minority-class recall particularly difficult.

---

## Project Structure

```
├── dataset/
│   ├── NBA_Train.csv
│   ├── NBA_Validation.csv
│   ├── NBA_Test.csv
│   └── Data Exploration & Insights.docx
│
├── models/                          # From-scratch model implementations
│   ├── mlp_from_scratch.py          #   MLP (one-hidden-layer neural network)
│   ├── mlp_inference.py             #   MLP inference / loading utilities
│   ├── knn_baseline.py              #   K-Nearest Neighbors
│   └── knn_inference.py             #   KNN inference utilities
│
├── logistice regression.ipynb       # Logistic Regression (softmax, from scratch)
├── mlp_from_scratch.ipynb           # MLP training notebook
├── KNN_experiments.ipynb            # KNN hyperparameter experiments
├── Data processing.ipynb            # Data cleaning & preprocessing
├── Final_Project_EDA.ipynb          # Exploratory data analysis & visualizations
│
├── outputs/                         # Trained model artifacts & predictions
│   ├── mlp/                         #   MLP weights, predictions, training curves
│   ├── knn/                         #   KNN model, predictions, reports
│   └── LogisticRegression/          #   LR predictions, training curves
│
├── evaluation/
│   ├── evaluate_all.py              # Unified evaluation framework
│   └── results/                     #   Confusion matrices, metrics tables, plots
│
├── Streamlit App/                   # Interactive web application
│   ├── app.py                       #   Home page
│   ├── pages/
│   │   ├── 1_Data_Overview.py       #   Dataset exploration
│   │   ├── 2_Draft_Projection.py    #   Per-player predictions
│   │   ├── 3_Whatif_Simulator.py    #   Hypothetical prospect simulator
│   │   └── 4_Model_Evaluation.py   #   Model comparison dashboard
│   ├── models/                      #   Model code for inference
│   ├── outputs/                     #   Saved weights & preprocessing
│   └── requirements.txt
│
└── reports/
    ├── PAML_Final_Project.pdf       # Final project report (IEEE format)
    └── NBA Draft Prediction Project.pdf
```

---

## Models

All three classifiers are implemented **from scratch** using only NumPy for array math and Pandas for data loading.

### 1. Logistic Regression (Softmax)
- Multinomial softmax regression with weighted cross-entropy loss
- Mini-batch gradient descent with L2 regularization
- Grid search over learning rate, regularization strength, batch size, epochs, and class weighting

### 2. K-Nearest Neighbors (KNN)
- Custom KNN with Euclidean distance (batched for efficiency)
- Distance-weighted voting with balanced class vote weights
- ANOVA F-score feature weighting to emphasize discriminative features
- Validation-based k selection (k = 1, 3, 5, 7, 9, 11, 15, 21)

### 3. Multi-Layer Perceptron (MLP) ✓ *Selected Model*
- One hidden layer (ReLU activation, softmax output)
- Weighted cross-entropy and focal loss support
- Adam / SGD optimizer with L2 regularization and early stopping
- Hyperparameter grid search (hidden dim, learning rate, activation, class weight mode, loss function)
- Validation-tuned binary calibration and decision thresholds
- Two-stage mode option (binary drafted/undrafted → round classifier)

---

## Results

### Test Set Performance (Tuned Thresholds)

| Metric               | Logistic Regression |  KNN  | **MLP** ✓ |
| -------------------- | :-----------------: | :---: | :-------: |
| **Macro-F1**         |        0.443        | 0.470 | **0.531** |
| **Multiclass AUROC** |        0.942        | 0.722 | **0.952** |
| Accuracy             |        0.947        | 0.968 | **0.979** |
| Drafted-Any F1       |        0.228        | 0.271 | **0.398** |
| Drafted-Any Recall   |      **0.740**      | 0.560 |   0.640   |

The **MLP** was selected as the best model based on test AUROC (0.952) and macro-F1 (0.531), significantly outperforming both baselines on the drafted minority classes.

### Class Imbalance Handling
- **Weighted cross-entropy** adjusts loss to up-weight rare drafted classes
- **Focal loss** variant further emphasizes hard-to-classify examples
- **Validation-tuned decision thresholds** improve minority-class recall
- **Macro-F1 + drafted-F1** used for model selection instead of accuracy

---

## Streamlit App

An interactive web app for exploring predictions, data, and model performance.

### Pages

| Page                  | Description                                                                           |
| --------------------- | ------------------------------------------------------------------------------------- |
| **Home**              | Project overview and model performance summary                                        |
| **Data Overview**     | Class distributions, stat profiles by draft class, conference & recruiting breakdowns |
| **Draft Projection**  | Search any player from the 2021 test set and see the MLP's prediction                 |
| **What-if Simulator** | Design a hypothetical prospect — adjust stats with sliders and run a live prediction  |
| **Model Evaluation**  | Confusion matrices, per-class metrics, MLP training curves, probability distributions |

### Running the App

```bash
cd "Streamlit App"
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Getting Started

### Prerequisites

- Python 3.10+
- NumPy, Pandas, Matplotlib, Seaborn (for training & evaluation)
- Streamlit, Plotly (for the web app)

### Reproducing Model Training

Each model has a **Jupyter notebook** as the primary entrypoint for training, experimentation, and result inspection. Standalone Python scripts are also provided as an alternative for CLI-based training.

#### Jupyter Notebooks (Primary)

| Model                | Notebook                     |
| -------------------- | ---------------------------- |
| Logistic Regression  | `logistice regression.ipynb` |
| MLP                  | `mlp_from_scratch.ipynb`     |
| KNN                  | `KNN_experiments.ipynb`      |
| Data Preprocessing   | `Data processing.ipynb`      |
| EDA & Visualizations | `Final_Project_EDA.ipynb`    |

```bash
# Open any notebook
jupyter notebook "mlp_from_scratch.ipynb"
```

#### Python Scripts (Alternative)

```bash
# Train the MLP
python models/mlp_from_scratch.py \
    --train dataset/NBA_Train.csv \
    --validation dataset/NBA_Validation.csv \
    --test dataset/NBA_Test.csv \
    --output-dir outputs/mlp

# Train the KNN baseline
python models/knn_baseline.py \
    --train-path dataset/NBA_Train.csv \
    --val-path dataset/NBA_Validation.csv \
    --test-path dataset/NBA_Test.csv \
    --output-dir outputs/knn
```

### Running Evaluation

```bash
python evaluation/evaluate_all.py
# Outputs saved to evaluation/results/
```

---

## Reports

- [Final Project Report (IEEE format)](reports/PAML_Final_Project.pdf)
- [Project Presentation Slides](reports/NBA%20Draft%20Prediction%20Project.pdf)
