# End-to-End Retail Banking Credit Scoring System



📌 Project Overview

This repository contains a production-ready credit scoring pipeline designed to automate and optimize credit risk appraisal for retail banking. Using a historical dataset of 1,000 credit applicants, the system orchestrates a clean workflow: establishing a local relational SQLite database,  transforming complex categorical profiles via one-hot encoding, correcting scale variances using mathematical standardization, and deploying a tuned  Logistic Regression binary classifier to compute structural default probabilities.



📊 Performance Metrics & Business Alignment

- Overall Model Accuracy: 72.00%
- Primary Optimization Focus (Recall for Class 0): In credit underwriting, the cost of a False Negative   
(approving a high-risk applicant who subsequently defaults) exponentially outweighs the opportunity cost of a False Positive (wrongfully rejecting a low-risk applicant). Therefore, this architecture prioritizes the optimization of recall for the defaultclass (0), giving risk managers tighter control over portfolio asset quality.



🛠️ Architecture & Core Pipeline Modules

1. Data Ingestion Layer (`db_setup.py`): Ingests raw data via Pandas and pipes it directly into a local SQLite matrix (`bank_data.db`), mirroring corporate financial data warehouses.
2. Exploratory Scratchpad (`eda_analysis.ipynb`): Examines multi-collinearity via cross-feature Pearson correlation matrices   
plotted on custom Seaborn heatmaps, ensuring zero data corruption or missing elements (`.isnull().sum()`).
3. Statistical Normalization & Modeling (`train_model.py`):

- Applies an isolated `StandardScaler` mapping to numeric features (e.g., Duration, Amount) to secure equal weight distribution against massive scalar variance.
- Leverages `scikit-learn` Logistic Regression to calculate maximum-likelihood decision boundaries between reliable and unstable capital borrowers.
- Serializes trained weights and pipeline variables into production-ready binaries (`credit_scoring_model.pkl` and `data_scaler.pkl`) via `joblib`.



📈 Model Evaluation
Statistical evaluations, including a 2x2 Confusion Matrix (mapping true/false positives and negatives) and an ROC Curve 
(measuring macro-level discriminative power across variable decision thresholds), are exported directly upon pipeline execution into `model_evaluation_plots.png`.