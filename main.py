import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ===============================
# 1. LOAD DATASET
# ===============================
file_path = r"C:\Users\ASUS\Documents\project\anomaly\Fusion_Data.csv"
df = pd.read_csv(file_path)

print("Total rows:", df.shape[0])
print("Columns:", df.columns)

# ===============================
# 2. FIX LABELS
# 0 -> 0, non-zero -> 1
# ===============================
df["labels"] = (df["labels"] != 0).astype(int)
print("Final labels:", np.unique(df["labels"]))

# ===============================
# 3. FEATURES & TARGET
# ===============================
X = df.drop(columns=["labels", "timestamp"])
y = df["labels"]

# ===============================
# 4. TRAIN–TEST SPLIT (80–20)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 5. SCALING
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 6. MODELS
# ===============================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVC": SVC(kernel="rbf", probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# ===============================
# 7. TRAIN + METRICS
# ===============================
print("\nMODEL RESULTS\n")

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy :", round(acc, 4))
    print("AUC      :", round(auc, 4))
    print("Confusion Matrix:")
    print(cm)

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===============================
# 8. LINEAR REGRESSION (ANOMALY)
# ===============================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

scores = lin_reg.predict(X_test)

threshold = np.mean(scores) + 2 * np.std(scores)
y_pred_lr = (scores > threshold).astype(int)

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, scores)
cm_lr = confusion_matrix(y_test, y_pred_lr)

print("\nLinear Regression")
print("Accuracy :", round(acc_lr, 4))
print("AUC      :", round(auc_lr, 4))
print("Confusion Matrix:")
print(cm_lr)

# ---- ROC Curve ----
fpr_lr, tpr_lr, _ = roc_curve(y_test, scores)
plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f"AUC = {auc_lr:.2f}")
plt.plot([0, 1], [0, 1], linestyle=":")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
