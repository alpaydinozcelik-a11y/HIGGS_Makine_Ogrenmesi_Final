import pandas as pd
import numpy as np

print("Makine Öğrenmesi Final Projesi Başladı")
print("main.py çalışıyor")
import pandas as pd

columns = ["target"] + [f"feature_{i}" for i in range(1, 29)]

df = pd.read_csv(
    "data/HIGGS.csv.gz",
    compression="gzip",
    header=None,
    names=columns,
    nrows=100000
)

print("Veri boyutu:", df.shape)
print(df.head())

print("\nSınıf dağılımı:")
print(df["target"].value_counts())

df.to_csv("data/higgs_100k.csv", index=False)

print("\n100.000 satırlık veri kaydedildi: data/higgs_100k.csv")
from sklearn.preprocessing import MinMaxScaler

X = df.drop("target", axis=1)
y = df["target"]

# IQR ile aykırı değerleri sınır değerlerine çekme
X_capped = X.copy()

for column in X_capped.columns:
    Q1 = X_capped[column].quantile(0.25)
    Q3 = X_capped[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X_capped[column] = X_capped[column].clip(lower_bound, upper_bound)

print("\nAykırı değer sınırlandırma tamamlandı.")

# MinMaxScaler ile ölçekleme
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_capped)

print("Ölçekleme tamamlandı.")
print("Ölçeklenmiş veri boyutu:", X_scaled.shape)
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# En iyi 15 özelliği ANOVA F-score ile seçme
selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X_scaled, y)

selected_features = X.columns[selector.get_support()]
feature_scores = selector.scores_[selector.get_support()]

selected_df = pd.DataFrame({
    "feature": selected_features,
    "score": feature_scores
}).sort_values(by="score", ascending=False)

print("\nSeçilen en iyi 15 özellik:")
print(selected_df)

selected_df.to_csv("data/selected_features.csv", index=False)
print("\nSeçilen özellikler kaydedildi: data/selected_features.csv")
print("Seçilmiş veri boyutu:", X_selected.shape)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(
    data=selected_df,
    x="score",
    y="feature",
    color="steelblue"
)

plt.title("ANOVA F-score ile Seçilen En İyi 15 Özellik")
plt.xlabel("F-score")
plt.ylabel("Özellik")
plt.tight_layout()

plt.savefig("figures/selected_features.png", dpi=300)
plt.show()

print("\nÖzellik seçimi grafiği kaydedildi: figures/selected_features.png")
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import numpy as np

# Modelleme için hızlı alt örneklem
model_df = df.sample(n=20000, random_state=42, stratify=df["target"])

X_model = model_df.drop("target", axis=1)
y_model = model_df["target"]

models = {
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9, 11]
        }
    },
    "SVM": {
        "model": SVC(probability=True),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    },
    "MLP": {
        "model": MLPClassifier(max_iter=300, random_state=42),
        "params": {
            "model__hidden_layer_sizes": [(50,), (100,)],
            "model__activation": ["relu", "tanh"]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1]
        }
    }
}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

results = []

for model_name, item in models.items():
    print(f"\nModel çalışıyor: {model_name}")

    fold_no = 1

    for train_index, test_index in outer_cv.split(X_model, y_model):
        X_train = X_model.iloc[train_index]
        X_test = X_model.iloc[test_index]
        y_train = y_model.iloc[train_index]
        y_test = y_model.iloc[test_index]

        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("model", item["model"])
        ])

        grid_search = GridSearchCV(
            pipeline,
            item["params"],
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        results.append({
            "model": model_name,
            "fold": fold_no,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "best_params": grid_search.best_params_
        })

        print(f"Fold {fold_no} tamamlandı. ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

        fold_no += 1

results_df = pd.DataFrame(results)
print("\nNested CV sonuçları:")
print(results_df)

results_df.to_csv("data/model_results.csv", index=False)
print("\nModel sonuçları kaydedildi: data/model_results.csv")

summary_df = results_df.groupby("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].mean()
print("\nOrtalama model performansları:")
print(summary_df)

summary_df.to_csv("data/model_summary.csv")
print("\nÖzet sonuçlar kaydedildi: data/model_summary.csv")
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import numpy as np

# Modelleme için stratified 10.000 örnek alıyoruz
X_model, _, y_model, _ = train_test_split(
    X,
    y,
    train_size=10000,
    stratify=y,
    random_state=42
)

models = {
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9, 11]
        }
    },
    "SVM": {
        "model": SVC(),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    },
    "MLP": {
        "model": MLPClassifier(max_iter=300, random_state=42),
        "params": {
            "model__hidden_layer_sizes": [(50,), (100,)],
            "model__activation": ["relu", "tanh"]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="logloss", random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1]
        }
    }
}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

results = []

for model_name, item in models.items():
    print(f"\nModel çalışıyor: {model_name}")

    fold_no = 1

    for train_index, test_index in outer_cv.split(X_model, y_model):
        X_train = X_model.iloc[train_index]
        X_test = X_model.iloc[test_index]
        y_train = y_model.iloc[train_index]
        y_test = y_model.iloc[test_index]

        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("model", item["model"])
        ])

        grid_search = GridSearchCV(
            pipeline,
            item["params"],
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        if hasattr(best_model.named_steps["model"], "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
        else:
            y_score = best_model.decision_function(X_test)

results.append({
    "model": model_name,
    "fold": fold_no,
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_score),
    "best_params": grid_search.best_params_
})

                                   