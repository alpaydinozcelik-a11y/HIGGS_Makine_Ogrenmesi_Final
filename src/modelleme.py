import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/higgs_100k.csv")

X = df.drop("target", axis=1)
y = df["target"]

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
        "params": {"model__n_neighbors": [3, 5, 7, 9, 11]}
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
roc_data = []

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

        for true_value, score_value in zip(y_test, y_score):
            roc_data.append({
                "model": model_name,
                "fold": fold_no,
                "y_true": true_value,
                "y_score": score_value
            })

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

        print(f"Fold {fold_no} tamamlandı. ROC-AUC: {roc_auc_score(y_test, y_score):.4f}")

        fold_no += 1

results_df = pd.DataFrame(results)
results_df.to_csv("data/model_results.csv", index=False)

summary_df = results_df.groupby("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].mean()
summary_df.to_csv("data/model_summary.csv")

roc_df = pd.DataFrame(roc_data)
roc_df.to_csv("data/roc_data.csv", index=False)

print("\nOrtalama model performansları:")
print(summary_df)

print("\nDosyalar kaydedildi:")
print("\nOrtalama model performansları:")
print(summary_df)

print("\nDosyalar kaydedildi:")
print("data/model_results.csv")
print("data/model_summary.csv")
print("data/roc_data.csv")