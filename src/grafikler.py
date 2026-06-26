import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

summary_df = pd.read_csv("data/model_summary.csv")

summary_df = summary_df.rename(columns={"Unnamed: 0": "model"})

summary_df.set_index("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Modellerin Ortalama Performans Karşılaştırması")
plt.ylabel("Skor")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("figures/model_performance.png", dpi=300)
plt.show()

print("Model performans grafiği kaydedildi: figures/model_performance.png")