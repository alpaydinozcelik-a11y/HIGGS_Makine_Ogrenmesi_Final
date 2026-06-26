import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("figures", exist_ok=True)

def draw_flowchart(title, steps, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.set_title(title, fontsize=14, fontweight="bold")

    x_positions = [1, 3.2, 5.4, 7.6]
    y = 3

    for i, step in enumerate(steps):
        box = FancyBboxPatch(
            (x_positions[i] - 0.8, y - 0.5),
            1.6,
            1.0,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor="#DCEEFF"
        )
        ax.add_patch(box)

        ax.text(
            x_positions[i],
            y,
            step,
            ha="center",
            va="center",
            fontsize=9
        )

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.9, y),
                xytext=(x_positions[i] + 0.9, y),
                arrowprops=dict(arrowstyle="->", lw=2)
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


steps_a = [
    "Eğitim\nKatı",
    "Özellik\nSeçimi",
    "En iyi 15\nÖzellik",
    "Dış Test\nKatı"
]

steps_b = [
    "Eğitim\nKatı",
    "GridSearchCV\nInner 3-fold",
    "En iyi\nParametreler",
    "Outer 5-fold\nTest"
]

draw_flowchart(
    "Flowchart A: İç Döngüde Özellik Seçimi",
    steps_a,
    "figures/flowchart_feature_selection.png"
)

draw_flowchart(
    "Flowchart B: İç Döngüde Hiperparametre Araması",
    steps_b,
    "figures/flowchart_hyperparameter.png"
)

print("Flowchart görselleri oluşturuldu:")
print("figures/flowchart_feature_selection.png")
print("figures/flowchart_hyperparameter.png")