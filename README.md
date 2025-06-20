# 🏦 Credit Risk Modeling

This project showcases a practical end-to-end **credit risk modeling pipeline** using Lending Club data, with the ultimate goal of implementing a **complete Expected Loss (EL) model**. The current focus is on developing a robust and interpretable **Probability of Default (PD)** model using industry best practices.

> 💡 New: Includes a self-contained `CreditScorecard` class for generating scaled scorecards, assigning credit scores, and mapping scores to PD or approval decisions.

---

## 📌 Project Roadmap

The final goal is to build an end-to-end **Expected Loss (EL)** model:

### 🔁 End-to-End Pipeline (Target Scope)
- ✅ **Probability of Default (PD)** 
- 🔜 **Loss Given Default (LGD)** — *currently in progress*
- 🔜 **Exposure at Default (EAD)**
- 🔜 **Expected Loss (EL)** Calculation  
  
$$
\textrm{EL} = \textrm{PD} \times \textrm{LGD} \times \textrm{EAD}
$$

---

## 🔍 Objectives (Current Stage: PD Modeling)

- Build a **binary classification model** to predict loan default.
- Apply **domain-specific preprocessing** for **interpretability** and **regulatory alignment**.
- Carefully avoid **data leakage**, especially with known post-outcome features in the Lending Club dataset.

---

## ⚙️ Key Features & Methodology

### ✅ Feature Engineering

- All variables were transformed to **categorical bins** using both **fine classing** and **optimal binning** techniques.
- Applied **Weight of Evidence (WoE)** transformation with the [`optbinning`](https://gnpalencia.org/optbinning/) library to support interpretability and monotonicity.
- Created reusable helper functions for **custom binning**, **IV calculation**, and **WoE encoding**: `feature_utils.py`
- Binning is done **only using training data** to preserve the integrity of the validation.

### 🧮 Scorecard Construction

- A dedicated `CreditScorecard` class (`score_card_utils.py`) is provided to:
  - Scale logistic regression coefficients to a user-defined score range (e.g., 300–850)
  - Compute borrower-level credit scores from dummified input data
  - Map scores to predicted PD (or 1-PD) using logistic transformation
  - Evaluate approval and rejection rates at various score/probability thresholds

### ⚠️ Data Leakage Prevention

- Careful attention was paid to **data leakage issues**, especially those stemming from:
  - Temporal leakage
  - Outcome-related features (`recoveries`, `last_pymnt_d`, `total_rec_prncp`, etc.)
- Data splitting (train/test) was performed **before** binning and transformation.

### 📈 Modeling & Validation

- A custom logistic regression model was used as the baseline **interpretable classifier**: `models.py`
- Evaluation metrics include:
  - **AUC / ROC**
  - **KS statistic**
  - **Confusion Matrix**

---

## 💡 Highlights

- 🧩 **Modular design**: clean separation between data prep, modeling, and evaluation
- 🧠 **Interpretability-first**: WoE binning and logistic regression for transparency
- 🔐 **Data leakage safeguards**: inspired by real-world credit risk model validation
- 📈 **Industry-aligned metrics**: AUC, KS, lift, and monotonicity checks

---

## 🧰 Tools & Libraries

- Python 3.11
- Jupyter Notebooks
- `optbinning`, `pandas`, `numpy`, `scikit-learn`
- `matplotlib`, `seaborn`

---

## 📜 Disclaimer

This project is for educational purposes only. It uses **public Lending Club data** and simulates a credit risk modeling pipeline in a **non-production** environment. No personally identifiable information (PII) is included.

---

## 🛠️ Future Improvements

- Implement **LGD** and **EAD** models
- Explore **gradient boosting** models and compare with logistic regression


