#  Customer Churn Predictor

> An end-to-end machine learning web app predicting telecom customer churn — built with Python, trained on 7,043 records, deployed on Streamlit.

 **[Live Demo](https://your-streamlit-link.streamlit.app)** ← replace after deploying

---

##  Model Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
|  XGBoost (selected) | 74.88% | **0.8427** |
| LightGBM | 74.10% | 0.8396 |
| Random Forest | 78.85% | 0.8235 |

> XGBoost selected as best model based on ROC-AUC score.

---

##  Key Business Insights

- **Month-to-month contracts** churn at **42.7%** vs only 2.8% on two-year contracts
- **Fiber optic customers** churn at **41.9%** despite being premium customers
- **Contract type** is the single strongest predictor (42% feature importance in XGBoost)
- **Financial features** (MonthlyCharges, TotalCharges, tenure) dominate in LightGBM & Random Forest

---

##  App Features

- Real-time churn probability prediction with risk level indicator
- Key risk factor analysis per customer
- Interactive sidebar with full customer profile inputs
- Model performance metrics dashboard

---

##  Project Structure

```
customer-churn-predictor/
├── app.py                  # Streamlit web application
├── model.py                # Model training + comparison script
├── clean_churn.py          # Data cleaning script
├── churn_model.pkl         # Saved XGBoost model
├── feature_names.pkl       # Feature names for prediction
├── churn_clean.csv         # Cleaned dataset
├── confusion_matrices.png  # Model comparison visual
├── requirements.txt        # Dependencies
└── README.md
```

---

##  Tools & Skills

`Python` `XGBoost` `LightGBM` `Scikit-learn` `Pandas` `NumPy` `Streamlit` `Seaborn` `Matplotlib` `Joblib`

---

##  Dataset

Telco Customer Churn — 7,043 customers, 21 features
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 👩‍💻 Author

**Farwa Irfan** — Data Scientist & Analyst
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/farwa-irfan-372141212/)
[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat&logo=github)](https://github.com/farwairfan112-gif)
