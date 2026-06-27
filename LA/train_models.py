import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from catboost import CatBoostClassifier

# SETTINGS

DATA_PATH = "data/Crime_Data_from_2020_to_2024.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

print("CRIME PREDICTION MODEL TRAINING")
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# REMOVE RARE CLASSES

crime_counts = df["Crm Cd Desc"].value_counts()
MIN_SAMPLES = 35000
valid_classes = crime_counts[crime_counts >= MIN_SAMPLES].index
df = df[df["Crm Cd Desc"].isin(valid_classes)].copy()
print()
print("Remaining crime classes:", df["Crm Cd Desc"].nunique())
print("Dataset shape:", df.shape)

# DATE FEATURES

df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
df["Date Rptd"] = pd.to_datetime(df["Date Rptd"], errors="coerce")
df["YEAR"] = df["DATE OCC"].dt.year
df["MONTH"] = df["DATE OCC"].dt.month
df["DAY"] = df["DATE OCC"].dt.day
df["DAY_OF_WEEK"] = df["DATE OCC"].dt.dayofweek
df["WEEK_OF_YEAR"] = df["DATE OCC"].dt.isocalendar().week.astype(int)
df["QUARTER"] = df["DATE OCC"].dt.quarter

# TIME FEATURES

df["TIME OCC"] = df["TIME OCC"].fillna(0).astype(int)
df["HOUR"] = df["TIME OCC"].astype(str).str.zfill(4).str[:2].astype(int)
df["MINUTE"] = df["TIME OCC"].astype(str).str.zfill(4).str[2:].astype(int)

# PERIOD OF DAY


def get_period(hour):
    if hour < 6:
        return "Night"
    elif hour < 12:
        return "Morning"
    elif hour < 18:
        return "Day"
    return "Evening"


df["PERIOD"] = df["HOUR"].apply(get_period)

# WEEKEND

df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)

# REPORT DELAY

df["REPORT_DELAY"] = (df["Date Rptd"] - df["DATE OCC"]).dt.days
df["REPORT_DELAY"] = df["REPORT_DELAY"].clip(lower=0)

# AGE

df["Vict Age"] = df["Vict Age"].replace(-1, np.nan)
df["Vict Age"] = df["Vict Age"].fillna(df["Vict Age"].median())

# REMOVE INVALID COORDINATES

df = df.dropna(subset=["LAT", "LON"])
df = df[(df["LAT"] != 0) & (df["LON"] != 0)]

# FILL CATEGORICAL FEATURES

categorical_columns = [
    "AREA NAME",
    "Vict Sex",
    "Vict Descent",
    "Premis Desc",
    "Weapon Desc",
]
for col in categorical_columns:
    df[col] = df[col].fillna("Unknown").astype(str)

# FEATURES

FEATURES = [
    "AREA NAME",
    "Vict Age",
    "Vict Sex",
    "Vict Descent",
    "Premis Desc",
    "Weapon Desc",
    "LAT",
    "LON",
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "WEEK_OF_YEAR",
    "QUARTER",
    "HOUR",
    "MINUTE",
    "PERIOD",
    "IS_WEEKEND",
    "REPORT_DELAY",
]
TARGET = "Crm Cd Desc"

# NUMERIC FEATURES

numeric_features = [
    "Vict Age",
    "LAT",
    "LON",
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "WEEK_OF_YEAR",
    "QUARTER",
    "HOUR",
    "MINUTE",
    "IS_WEEKEND",
    "REPORT_DELAY",
]
for col in numeric_features:
    df[col] = df[col].fillna(df[col].median())

# PREPARE DATASET

dataset = df[FEATURES + [TARGET]].copy()
X = dataset[FEATURES]
y = dataset[TARGET]

# CATBOOST CATEGORICAL FEATURES

CAT_FEATURES = [
    "AREA NAME",
    "Vict Sex",
    "Vict Descent",
    "Premis Desc",
    "Weapon Desc",
    "PERIOD",
]
cat_features = [FEATURES.index(col) for col in CAT_FEATURES]

# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE, shuffle=True
)

print()
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))
print("Crime classes    :", y.nunique())

# MODEL 1
# CRIME TYPE PREDICTION

print()
print("MODEL 1 - CRIME TYPE PREDICTION")

crime_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=8,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    early_stopping_rounds=70,
    verbose=100,
)

crime_model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True,
)

# PREDICTION

print()
print("Predicting...")

y_pred = crime_model.predict(X_test)
y_pred = y_pred.flatten()
y_prob = crime_model.predict_proba(X_test)

# METRICS

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print()
print("RESULTS")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print()
print(classification_report(y_test, y_pred, zero_division=0))

# CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=crime_model.classes_, columns=crime_model.classes_)
cm_df.to_csv(MODEL_DIR / "crime_confusion_matrix.csv", encoding="utf-8-sig")

# FEATURE IMPORTANCE

importance = pd.DataFrame(
    {"Feature": FEATURES, "Importance": crime_model.get_feature_importance()}
)
importance = importance.sort_values("Importance", ascending=False)
importance.to_csv(
    MODEL_DIR / "crime_feature_importance.csv", index=False, encoding="utf-8-sig"
)

print()
print("Top 10 important features:")
print(importance.head(10))

# CROSS VALIDATION METRICS

results = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Value": [accuracy, precision, recall, f1],
    }
)
results.to_csv(MODEL_DIR / "crime_metrics.csv", index=False, encoding="utf-8-sig")

# SAVE

joblib.dump(crime_model, MODEL_DIR / "crime_type_model.pkl")
joblib.dump(sorted(y.unique().tolist()), MODEL_DIR / "crime_classes.pkl")
joblib.dump(FEATURES, MODEL_DIR / "crime_features.pkl")
print("Crime Type model training completed.")

# MODEL 2
# AREA RISK PREDICTION

print()
print("MODEL 2 - AREA RISK PREDICTION")

# CREATE RISK LEVELS

risk_df = df.copy()
risk_counts = risk_df["AREA NAME"].value_counts()
high_thr = risk_counts.quantile(0.75)
medium_thr = risk_counts.quantile(0.50)
low_thr = risk_counts.quantile(0.25)
risk_map = {}
for area, count in risk_counts.items():
    if count >= high_thr:
        risk_map[area] = "Very High"
    elif count >= medium_thr:
        risk_map[area] = "High"
    elif count >= low_thr:
        risk_map[area] = "Medium"
    else:
        risk_map[area] = "Low"
risk_df["RISK_LEVEL"] = risk_df["AREA NAME"].map(risk_map)

# DATA

RISK_TARGET = "RISK_LEVEL"
risk_dataset = risk_df[FEATURES + [RISK_TARGET]].copy()
X_risk = risk_dataset[FEATURES]
y_risk = risk_dataset[RISK_TARGET]

# TRAIN TEST

X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
    X_risk,
    y_risk,
    test_size=0.20,
    stratify=y_risk,
    random_state=RANDOM_STATE,
    shuffle=True,
)

# MODEL

risk_model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    early_stopping_rounds=100,
    verbose=100,
)

risk_model.fit(
    X_train_risk,
    y_train_risk,
    cat_features=cat_features,
    eval_set=(X_test_risk, y_test_risk),
    use_best_model=True,
)

# PREDICTION

risk_pred = risk_model.predict(X_test_risk)
risk_pred = risk_pred.flatten()
risk_prob = risk_model.predict_proba(X_test_risk)

# METRICS

risk_accuracy = accuracy_score(y_test_risk, risk_pred)
risk_precision = precision_score(
    y_test_risk, risk_pred, average="weighted", zero_division=0
)
risk_recall = recall_score(y_test_risk, risk_pred, average="weighted", zero_division=0)
risk_f1 = f1_score(y_test_risk, risk_pred, average="weighted", zero_division=0)

print()
print("AREA RISK RESULTS")
print(f"Accuracy : {risk_accuracy:.4f}")
print(f"Precision: {risk_precision:.4f}")
print(f"Recall   : {risk_recall:.4f}")
print(f"F1 Score : {risk_f1:.4f}")
print(classification_report(y_test_risk, risk_pred, zero_division=0))

# FEATURE IMPORTANCE

risk_importance = pd.DataFrame(
    {"Feature": FEATURES, "Importance": risk_model.get_feature_importance()}
)
risk_importance = risk_importance.sort_values("Importance", ascending=False)
risk_importance.to_csv(
    MODEL_DIR / "risk_feature_importance.csv", index=False, encoding="utf-8-sig"
)

print()
print("Most important features:")
print(risk_importance.head(10))

# CONFUSION MATRIX

risk_cm = confusion_matrix(y_test_risk, risk_pred)
risk_cm_df = pd.DataFrame(
    risk_cm, index=risk_model.classes_, columns=risk_model.classes_
)
risk_cm_df.to_csv(MODEL_DIR / "risk_confusion_matrix.csv", encoding="utf-8-sig")

# SAVE METRICS

risk_metrics = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Value": [risk_accuracy, risk_precision, risk_recall, risk_f1],
    }
)
risk_metrics.to_csv(MODEL_DIR / "risk_metrics.csv", index=False, encoding="utf-8-sig")

# SAVE MODEL

joblib.dump(risk_model, MODEL_DIR / "area_risk_model.pkl")
print()
print("Area Risk training completed.")

# MODEL 3
# CRIME HOTSPOT PREDICTION

print()
print("MODEL 3 - CRIME HOTSPOT PREDICTION")
hotspot_df = df.copy()

# CREATE GRID

LAT_BINS = 30
LON_BINS = 30
hotspot_df["LAT_GRID"] = pd.cut(hotspot_df["LAT"], bins=LAT_BINS, labels=False)
hotspot_df["LON_GRID"] = pd.cut(hotspot_df["LON"], bins=LON_BINS, labels=False)
hotspot_df["GRID_ID"] = (
    hotspot_df["LAT_GRID"].astype(str) + "_" + hotspot_df["LON_GRID"].astype(str)
)

# HOTSPOT LABEL

grid_counts = hotspot_df["GRID_ID"].value_counts()
threshold = grid_counts.quantile(0.80)
hotspot_df["HOTSPOT"] = hotspot_df["GRID_ID"].map(
    lambda x: 1 if grid_counts[x] >= threshold else 0
)

# DATA

HOTSPOT_TARGET = "HOTSPOT"
hotspot_dataset = hotspot_df[FEATURES + [HOTSPOT_TARGET]].copy()
X_hot = hotspot_dataset[FEATURES]
y_hot = hotspot_dataset[HOTSPOT_TARGET]

# TRAIN TEST

X_train_hot, X_test_hot, y_train_hot, y_test_hot = train_test_split(
    X_hot,
    y_hot,
    stratify=y_hot,
    test_size=0.20,
    random_state=RANDOM_STATE,
    shuffle=True,
)

# MODEL

hotspot_model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="Accuracy",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    early_stopping_rounds=100,
    verbose=100,
)

hotspot_model.fit(
    X_train_hot,
    y_train_hot,
    cat_features=cat_features,
    eval_set=(X_test_hot, y_test_hot),
    use_best_model=True,
)

# PREDICTION

hot_pred = hotspot_model.predict(X_test_hot)
hot_pred = hot_pred.flatten()
hot_prob = hotspot_model.predict_proba(X_test_hot)

# METRICS

hot_accuracy = accuracy_score(y_test_hot, hot_pred)
hot_precision = precision_score(y_test_hot, hot_pred, average="binary", zero_division=0)
hot_recall = recall_score(y_test_hot, hot_pred, average="binary", zero_division=0)
hot_f1 = f1_score(y_test_hot, hot_pred, average="binary", zero_division=0)

print()
print("HOTSPOT RESULTS")
print(f"Accuracy : {hot_accuracy:.4f}")
print(f"Precision: {hot_precision:.4f}")
print(f"Recall   : {hot_recall:.4f}")
print(f"F1 Score : {hot_f1:.4f}")
print(classification_report(y_test_hot, hot_pred, zero_division=0))

# FEATURE IMPORTANCE

hot_importance = pd.DataFrame(
    {"Feature": FEATURES, "Importance": hotspot_model.get_feature_importance()}
)
hot_importance = hot_importance.sort_values("Importance", ascending=False)
hot_importance.to_csv(
    MODEL_DIR / "hotspot_feature_importance.csv", index=False, encoding="utf-8-sig"
)
print()
print("Top hotspot features:")
print(hot_importance.head(10))

# CONFUSION MATRIX

hot_cm = confusion_matrix(y_test_hot, hot_pred)
hot_cm_df = pd.DataFrame(
    hot_cm, index=["Not Hotspot", "Hotspot"], columns=["Not Hotspot", "Hotspot"]
)
hot_cm_df.to_csv(MODEL_DIR / "hotspot_confusion_matrix.csv", encoding="utf-8-sig")

# METRICS

hot_metrics = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Value": [hot_accuracy, hot_precision, hot_recall, hot_f1],
    }
)
hot_metrics.to_csv(MODEL_DIR / "hotspot_metrics.csv", index=False, encoding="utf-8-sig")

# SAVE MODEL

joblib.dump(hotspot_model, MODEL_DIR / "hotspot_model.pkl")
print()
print("Hotspot training completed.")

# TOP 10 CRIMES

print()
print("TOP 10 CRIMES")
top10 = df["Crm Cd Desc"].value_counts().head(10)
top10_df = pd.DataFrame({"Crime": top10.index, "Count": top10.values})
print(top10_df)
top10_df.to_csv(MODEL_DIR / "top10_crimes.csv", index=False, encoding="utf-8-sig")

# AREA CRIME RANKING

print()
print("AREA CRIME RANKING")
area_ranking = (
    df.groupby("AREA NAME")
    .size()
    .reset_index(name="Crime Count")
    .sort_values("Crime Count", ascending=False)
    .reset_index(drop=True)
)
area_ranking["Rank"] = np.arange(1, len(area_ranking) + 1)
area_ranking = area_ranking[["Rank", "AREA NAME", "Crime Count"]]
print(area_ranking)
area_ranking.to_csv(MODEL_DIR / "area_ranking.csv", index=False, encoding="utf-8-sig")

# AREA STATISTICS

area_statistics = (
    df.groupby("AREA NAME")
    .agg(
        Crimes=("Crm Cd Desc", "count"),
        Average_Age=("Vict Age", "mean"),
        Latitude=("LAT", "mean"),
        Longitude=("LON", "mean"),
    )
    .reset_index()
)
area_statistics.to_csv(
    MODEL_DIR / "area_statistics.csv", index=False, encoding="utf-8-sig"
)

# SAVE

metadata = {
    "features": FEATURES,
    "target": TARGET,
    "categorical_features": CAT_FEATURES,
    "areas": sorted(df["AREA NAME"].dropna().unique().tolist()),
    "premises": sorted(df["Premis Desc"].dropna().unique().tolist()),
    "weapons": sorted(df["Weapon Desc"].dropna().unique().tolist()),
    "victim_sex": sorted(df["Vict Sex"].dropna().unique().tolist()),
    "victim_descent": sorted(df["Vict Descent"].dropna().unique().tolist()),
    "crime_classes": sorted(df["Crm Cd Desc"].dropna().unique().tolist()),
    "risk_levels": ["Low", "Medium", "High", "Very High"],
    "training_rows": len(df),
    "training_features": len(FEATURES),
    "crime_classes_count": df["Crm Cd Desc"].nunique(),
    "created_models": [
        "crime_type_model.pkl",
        "area_risk_model.pkl",
        "hotspot_model.pkl",
    ],
}
joblib.dump(metadata, MODEL_DIR / "metadata.pkl")
joblib.dump(FEATURES, MODEL_DIR / "features.pkl")
joblib.dump(CAT_FEATURES, MODEL_DIR / "categorical_features.pkl")
joblib.dump(sorted(df["AREA NAME"].unique().tolist()), MODEL_DIR / "areas.pkl")
joblib.dump(sorted(df["Premis Desc"].unique().tolist()), MODEL_DIR / "premises.pkl")
joblib.dump(sorted(df["Weapon Desc"].unique().tolist()), MODEL_DIR / "weapons.pkl")
joblib.dump(sorted(df["Vict Sex"].unique().tolist()), MODEL_DIR / "victim_sex.pkl")
joblib.dump(
    sorted(df["Vict Descent"].unique().tolist()), MODEL_DIR / "victim_descent.pkl"
)

# TRAINING SUMMARY

print()
print("TRAINING SUMMARY")
print(f"Dataset size      : {len(df):,}")
print(f"Features          : {len(FEATURES)}")
print(f"Crime classes     : {df['Crm Cd Desc'].nunique()}")
print(f"Areas             : {df['AREA NAME'].nunique()}")
print(f"Premises          : {df['Premis Desc'].nunique()}")
print(f"Weapons           : {df['Weapon Desc'].nunique()}")

print()
print("ALL MODELS TRAINED SUCCESSFULLY")
