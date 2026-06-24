import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# 1. DATA PREPARATION
print("--- Step 1-2: Loading data ---")
df = pd.read_csv("brent_prices.csv", sep=",")
df = df.dropna()

df["price_diff"] = df["close"] - df["close"].shift(1)
df["Target"] = np.where(df["price_diff"] > 0, 1, 0)

df["prev_close"] = df["close"].shift(1)
df["prev_volume"] = df["volume"].shift(1)
df = df.dropna()

X = df[["prev_close", "prev_volume"]]
y = df["Target"]

# 4. MODEL TRAINING
print("--- Step 4: Training model ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")

# 5. MODEL EVALUATION
print("--- Step 5: Evaluation ---")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nSaving matrix to file...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Down (0)", "Up (1)"],
    yticklabels=["Down (0)", "Up (1)"],
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix_task2.png", dpi=300)
plt.close()
print("Done! File 'confusion_matrix_task2.png' saved.")
