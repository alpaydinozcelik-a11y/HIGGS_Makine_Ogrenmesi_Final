import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

os.makedirs("figures", exist_ok=True)

roc_df = pd.read_csv("data/roc_data.csv")

# Class 1 için OVA ROC
plt.figure(figsize=(8, 6))

for model_name in roc_df["model"].unique():
    model_data = roc_df[roc_df["model"] == model_name]

    fpr, tpr, _ = roc_curve(
        model_data["y_true"],
        model_data["y_score"]
    )

    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} Class 1 (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.title("OVA ROC Eğrileri - Class 1")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/roc_ova_class1.png", dpi=300)
plt.close()


# Class 0 için OVA ROC
plt.figure(figsize=(8, 6))

for model_name in roc_df["model"].unique():
    model_data = roc_df[roc_df["model"] == model_name]

    y_true_class0 = 1 - model_data["y_true"]
    y_score_class0 = -model_data["y_score"]

    fpr, tpr, _ = roc_curve(
        y_true_class0,
        y_score_class0
    )

    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} Class 0 (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.title("OVA ROC Eğrileri - Class 0")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/roc_ova_class0.png", dpi=300)
plt.close()

print("OVA ROC grafikleri kaydedildi:")
print("figures/roc_ova_class1.png")
print("figures/roc_ova_class0.png")