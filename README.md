# Loan Approval Prediction Pipeline

A machine learning pipeline that predicts whether a loan will be approved
or rejected based on applicant financial information.
The project uses the Kaggle Loan Approval Dataset and tracks multiple
experiments using MLflow to observe how different features affect prediction accuracy.

---

## Dataset

The choice of dataset was influenced by the main plan to handle loans and banking 
transactions using machine learning for faster and accurate process.

**Source:** Kaggle — Loan Approval Prediction Dataset  
**Rows:** 4269 applicants  
**Columns:** 13 features including income, loan amount, credit score, assets, education, employment status and loan term.  
**Target:** loan_status — Approved or Rejected  
**Dataset link:** https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

---

## Technologies Used

- Python 3.12
- Pandas — data loading and cleaning
- Scikit-learn — model training
- MLflow — experiment tracking
- Git and GitHub — version control

---

## How to Install

1. Clone the repository
git clone https://github.com/Princedwane/loan-prediction-pipeline.git

2. Create virtual environment
python3 -m venv venv

3. Activate it
source venv/bin/activate

4. Install dependencies
pip install -r requirements.txt

---

## How to Run

Run the pipeline:
python mini_project1.py

Check experiment results:
python check_runs.py

---

## Experiment Results

| Run | Features | Test Size | Accuracy |
|---|---|---|---|
| Run 1 Basic | income, loan amount, cibil score | 0.3 | 73.9% |
| Run 2 Mid | loan term, dependents, education | 0.3 | 62.7% |
| Run 3 All features | All 11 columns | 0.3 | 70.4% |
| Run 4 All features | All 11 columns | 0.5 | 72.0% |

---

## Insights

- Run 1 performed best using only 3 features — income, loan amount
  and credit score. These are the most financially relevant factors
  for loan approval decisions.Banks primarily look at income and credit score so these 3 features capture the most important patterns.

- Run 2 was the weakest at 62.7% because loan term, dependents and
  education alone do not provide enough financial evidence for the model.They are weak predictors thus dont provide enough evidence for model to decide approval confidently

- Adding more features in Run 3 did not beat Run 1. Feature quality 
  matters more than quantity — irrelevant features can confuse the model.
  
- Run 4 used 50% for testing instead of 30% and got 72% vs 70.4% in Run 3.
  This is likely random variation from the shuffle, not a real improvement.

- Credit score (cibil_score) is the most important predictor.
  Any run without it performed poorly.

- The dataset had no missing values — remove_missing found nothing to drop.

---

## Project Structure

mini_project1.py — main pipeline with 4 functions and 4 MLflow runs
check_runs.py — displays all experiment results
README.md — project documentation
requirements.txt — project dependencies

---

## Author

Prince John  — Information security & Data Science Student - UNIQUE ACADEMY, Tanzania.
GitHub: https://github.com/Princedwane