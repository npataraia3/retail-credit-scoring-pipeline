import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
import joblib

url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv(url)

connection = sqlite3.connect("bank_data.db")

df.to_sql("applicants", connection, if_exists="replace", index=False)

categories = [
    'status', 'credit_history', 'purpose', 'savings', 
    'employment_duration', 'personal_status_sex', 
    'other_debtors', 'property', 'other_installment_plans', 
    'housing', 'job', 'telephone', 'foreign_worker'
]

df_encoded = pd.get_dummies(df, columns=categories, drop_first=True)

X = df_encoded.drop(columns=['credit_risk'])
y = df_encoded['credit_risk']

numerical_cols = [
    'duration', 'amount', 'installment_rate',
    'present_residence', 'age', 'number_credits', 'people_liable'
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.2%}\n")
print(classification_report(y_test, predictions))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix (Good vs Bad Loans)')

RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax2)
ax2.set_title('ROC Curve (Model Discriminative Power)')

plt.tight_layout()
plt.savefig('model_evaluation.png')

joblib.dump(model, 'credit_scoring_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
