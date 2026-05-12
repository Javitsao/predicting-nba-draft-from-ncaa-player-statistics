# NBA Draft Predictor — Streamlit App

An interactive web app that predicts whether an NCAA college basketball player will be selected in the NBA Draft (Undrafted / 2nd Round / 1st Round).

---

## How to Run

```bash
pip install streamlit pandas numpy plotly
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## App Pages

| Page | Description |
|---|---|
| **Home** | Project overview and model performance summary |
| **Data Overview** | Class distributions, stat profiles by draft class, recruiting rank and conference breakdowns |
| **Draft Projection** | Search any player from the 2020–2021 test set — see MLP predicted draft class, confidence, and full stat profile |
| **What-if Simulator** | Choose a player template, adjust stats with sliders, and run a live MLP prediction |
| **Model Evaluation** | Confusion matrices, per-class metrics, MLP training curves, and probability distributions |

---

## File Structure

```
Streamlit app/
├── app.py
├── utils.py
├── requirements.txt
├── pages/
│   ├── 1_Data_Overview.py
│   ├── 2_Draft_Projection.py
│   ├── 3_Whatif_Simulator.py
│   └── 4_Model_Evaluation.py
├── models/
│   ├── mlp_from_scratch.py
│   └── mlp_inference.py
├── dataset/
│   ├── NBA_Train.csv
│   └── NBA_Test.csv
├── outputs/
│   ├── mlp/
│   │   ├── mid_checkpoint/
│   │   │   ├── mlp_model.npz
│   │   │   ├── mlp_preprocessing.json
│   │   │   ├── mlp_results.json
│   │   │   └── mlp_predictions_test.csv
│   │   └── mlp_training_validation_loss.csv
│   ├── knn/
│   │   └── knn_predictions_test.csv
│   └── LogisticRegression/
│       └── lr_test_predictions.csv
└── images/
    └── nba_logo.svg
```
