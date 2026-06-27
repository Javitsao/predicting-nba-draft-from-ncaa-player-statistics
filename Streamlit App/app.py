import streamlit as st
from pathlib import Path
from utils import setup_logo

st.set_page_config(
    page_title='NBA Draft Predictor',
    page_icon='🏀',
    layout='wide'
)

setup_logo()

st.title('🏀 NBA Draft Predictor')

st.markdown('''
Predict whether an NCAA basketball player will be selected in the **NBA Draft** —
trained on 2009–2021 college basketball statistics from the Kaggle
*College Basketball 2009–2021 + NBA Advanced Stats* dataset.
''')

st.markdown('---')

col1, col2 = st.columns(2)

with col1:
    st.markdown('''
### Navigate
Use the sidebar to access:

- **Data Overview** — Explore the dataset: class distributions, feature statistics, and draft trends
- **Draft Projection** — Search any player from our 2020–2021 test set to see the MLP's draft prediction
- **What-if Simulator** — Design a hypothetical prospect and explore how stats affect draft probability
- **Model Evaluation** — Confusion matrices, training curves, and metric comparisons across all three models
''')

with col2:
    st.markdown('### Model Performance')
    st.markdown('''
We trained three classifiers **from scratch** (NumPy only) and selected the best based on **Test AUROC**:

| Model | Test Macro-F1 | Test AUROC |
|---|---|---|
| Logistic Regression | 0.417 | — |
| KNN (k=3) | 0.474 | 0.722 |
| **MLP (selected)** ✓ | **0.527** | **0.959** |

The MLP uses one hidden layer (16 units, ReLU) with weighted cross-entropy to handle
the severe class imbalance (>96% of players go undrafted).
''')

st.markdown('---')
st.markdown('''
**Target classes:**  `0` Undrafted · `1` 1st Round · `2` 2nd Round
**Features:** 63 numeric + 3 categorical (team, conference, role)
**Data split:** Time-aware — train 2009–2017, validation 2018–2019, test 2020–2021
**Team:** Cornell Tech PAML Final Project
''')
