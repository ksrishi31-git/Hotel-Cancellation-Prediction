
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
sns.set_style("whitegrid")
df = pd.read_csv("hotel_bookings.csv")
print("\n===== FIRST 5 ROWS OF DATASET =====")
print(df.head().to_string())
missing = df.isnull().sum()
missing = missing[missing > 0]
plt.figure(figsize=(7, 4))
missing.plot(kind="bar", color="steelblue")
plt.title("Missing Values Before Cleaning")
plt.xlabel("Columns")
plt.ylabel("Missing Values")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
df = df.drop_duplicates().copy()
df.drop(columns=[
    "company",
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list"
], inplace=True)
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df["agent"] = df["agent"].fillna(-1).astype(str)
df["total_guests"] = (
    df["adults"] + df["children"] + df["babies"]
)
df["total_nights"] = (
    df["stays_in_week_nights"] +
    df["stays_in_weekend_nights"]
)
df["lead_time_log"] = np.log1p(df["lead_time"])
df["total_previous_bookings"] = (
    df["previous_cancellations"] +
    df["previous_bookings_not_canceled"]
)
df["has_children"] = (
    (df["children"] + df["babies"]) > 0
).astype(int)
df["is_long_stay"] = (
    df["total_nights"] >= 7
).astype(int)
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)
num_cols = X.select_dtypes(
    include=np.number
).columns.tolist()
cat_cols = X.select_dtypes(
    exclude=np.number
).columns.tolist()
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ]), cat_cols)
])
best_accuracy = -1
best_C = 1.0
best_weight = None
best_threshold = 0.5
for C in [0.01, 0.1, 1, 10, 100]:
    for weight in [None, "balanced"]:
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                C=C,
                max_iter=3000,
                solver="lbfgs",
                class_weight=weight,
                random_state=42
            ))
        ])
        model.fit(X_tr, y_tr)
        val_prob = model.predict_proba(X_val)[:, 1]
        for threshold in np.arange(0.30, 0.751, 0.01):
            val_pred = (
                val_prob >= threshold
            ).astype(int)
            score = accuracy_score(y_val, val_pred)
            if score > best_accuracy:
                best_accuracy = score
                best_C = C
                best_weight = weight
                best_threshold = float(threshold)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        C=best_C,
        max_iter=3000,
        solver="lbfgs",
        class_weight=best_weight,
        random_state=42
    ))
])
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (
    y_prob >= best_threshold
).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
print("\n==================================")
print("          MODEL RESULTS")
print("==================================")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Cancelled", "Cancelled"],
    yticklabels=["Not Cancelled", "Cancelled"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    color="blue",
    linewidth=2,
    label=f"Logistic Regression (AUC = {auc:.2f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    "r--",
    label="Random Classifier"
)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()
sample = X_test.sample(
    n=10,
    random_state=25
)
probabilities = model.predict_proba(sample)[:, 1]
predictions = (
    probabilities >= best_threshold
).astype(int)
examples = sample[[
    "hotel",
    "lead_time",
    "adults",
    "total_nights",
    "deposit_type"
]].copy()
examples["Actual Status"] = (
    y_test.loc[sample.index].map({
        0: "Not Cancelled",
        1: "Cancelled"
    })
)
examples["Predicted Status"] = [
    "Cancelled" if p == 1 else "Not Cancelled"
    for p in predictions
]
examples["Probability"] = [
    f"{p * 100:.2f}%"
    for p in probabilities
]
examples["Result"] = [
    "Correct" if actual == pred else "Incorrect"
    for actual, pred in zip(
        y_test.loc[sample.index],
        predictions
    )
]
print("\n===== REAL HOTEL BOOKING PREDICTIONS =====")
print(examples.to_string(index=False))
correct = (
    examples["Result"] == "Correct"
).sum()
print(f"\nCorrect Predictions: {correct}/10")
print("\nProject Completed Successfully!")
