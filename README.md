# 💳 Credit Card Fraud Detection with Imbalanced Data Handling

## 📌 Problem Statement

Fraud detection is a classic example of an **extremely imbalanced classification problem**, where fraudulent transactions represent only ~1–2% of total data.

In such cases, accuracy becomes misleading. A model predicting "no fraud" for every transaction would still achieve 98%+ accuracy.

This project focuses on building and evaluating models using appropriate metrics and threshold tuning strategies.

---

## 📊 Dataset

Synthetic dataset generated using `make_classification` with:

- 10,000 transactions
- 10 features
- 1.6% fraud rate
- Severe class imbalance (≈ 98% normal, 2% fraud)

---

## ⚙️ Approach

1. Generated highly imbalanced dataset
2. Applied feature scaling using `StandardScaler`
3. Compared two models:
   - Logistic Regression
   - Random Forest
4. Evaluated using:
   - ROC-AUC
   - Precision-Recall Curve
   - F1-score
5. Tuned classification threshold to optimize fraud detection performance

---

## 📈 Model Performance

### Logistic Regression
- ROC-AUC: 0.69
- Poor precision on fraud class
- High false positives

### Random Forest (Default Threshold = 0.50)
- ROC-AUC: 0.81
- Fraud F1-score: 0.31
- Low recall on fraud

### Random Forest (Tuned Threshold = 0.13)
- Fraud F1-score improved to **0.56**
- Fraud Precision: 0.78
- Fraud Recall: 0.44

Threshold tuning significantly improved fraud detection performance without drastically increasing false positives.

---

## 📉 Key Insights

- Accuracy is not reliable for imbalanced datasets
- Precision-Recall curve is more informative than ROC in rare-event detection
- Default threshold (0.5) is not always optimal
- Threshold tuning can dramatically improve real-world fraud detection systems

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt