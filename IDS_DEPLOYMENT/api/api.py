from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import shap
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="Deep Learning IDS API")


# -----------------------------
# INPUT SCHEMA
# -----------------------------
class FeatureInput(BaseModel):
    features: list
    explain: bool = True


# -----------------------------
# CUSTOM ATTENTION LAYER
# -----------------------------
@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):

        score = tf.matmul(inputs, self.W)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)

        return context


# -----------------------------
# PATHS
# -----------------------------
MODEL_PATH = r"D:\My_Space\College\6th_SEM\Deep_Learning_Project\IDS_DEPLOYMENT\models\bilstm_attention_model.keras"
SCALER_PATH = r"D:\My_Space\College\6th_SEM\Deep_Learning_Project\IDS_DEPLOYMENT\models\scaler_attention.pkl"
DATASET_PATH = r"D:\My_Space\College\6th_SEM\Deep_Learning_Project\IDS\Data\processed_03\cicids2017_merged.csv"


# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"Attention": Attention}
)

scaler = joblib.load(SCALER_PATH)

EXPECTED_FEATURES = scaler.n_features_in_

df = pd.read_csv(DATASET_PATH)

feature_names = df.drop(columns=["Label"]).columns.tolist()

print("Model loaded successfully")
print("Expected features:", EXPECTED_FEATURES)


# -----------------------------
# MODEL PREDICT
# -----------------------------
def model_predict(data):

    data_scaled = scaler.transform(data)

    data_seq = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)

    return model.predict(data_seq, verbose=0)


# -----------------------------
# SHAP EXPLAINER
# -----------------------------
background = np.zeros((1, EXPECTED_FEATURES))

explainer = shap.KernelExplainer(model_predict, background)


# -----------------------------
# LABEL MAPPING (FIXED)
# -----------------------------
def map_label(label_str):

    l = str(label_str).strip().upper()

    if l == "BENIGN":
        return 0

    elif "BOT" in l:
        return 1

    elif "DDOS" in l:
        return 2

    elif "DOS" in l and "DDOS" not in l:
        return 3

    elif "PATATOR" in l or "FTP" in l or "SSH" in l:
        return 4

    elif "PORTSCAN" in l:
        return 5

    elif "WEB" in l or "INFILTRATION" in l:
        return 6

    return 0


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def health():
    return {"status": "IDS API running"}


# -----------------------------
# GET RANDOM SAMPLE
# -----------------------------
@app.get("/get_sample")
def get_sample():

    try:

        sample = df.sample(1)

        label_col = "Label"

        actual_label = map_label(sample[label_col].values[0])

        X_df = sample.drop(columns=[label_col]).select_dtypes(include=[np.number])

        features = X_df.iloc[:, :EXPECTED_FEATURES].values.flatten().tolist()

        return {
            "features": features,
            "actual_label": actual_label
        }

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# PREDICTION
# -----------------------------
@app.post("/predict")
def predict(data: FeatureInput):

    try:

        features = data.features

        if len(features) != EXPECTED_FEATURES:

            return {"error": f"Expected {EXPECTED_FEATURES}, got {len(features)}"}

        X = np.array(features).reshape(1, -1)

        pred = model_predict(X)

        attack_class = int(np.argmax(pred))

        result = {
            "prediction": attack_class,
            "probabilities": pred[0].tolist(),
            "top_features": [],
            "importance_scores": []
        }

        # SHAP (optional)
        if data.explain:

            shap_values = explainer.shap_values(X, nsamples=40)

            if isinstance(shap_values, list):

                importance = np.abs(shap_values[attack_class][0])

            else:

                importance = np.abs(np.array(shap_values).flatten()[:EXPECTED_FEATURES])

            top_idx = np.argsort(importance)[-5:][::-1]

            result["top_features"] = [feature_names[i] for i in top_idx]

            result["importance_scores"] = importance[top_idx].tolist()

        return result

    except Exception as e:

        import traceback

        return {"error": str(e), "trace": traceback.format_exc()}


# -----------------------------
# MODEL EVALUATION
# -----------------------------
@app.get("/evaluate")
def evaluate_model():

    try:

        if len(df) > 10000:

            eval_df = df.sample(10000, random_state=42)

        else:

            eval_df = df

        y_raw = eval_df["Label"].values

        y = np.array([map_label(l) for l in y_raw])

        X_df = eval_df.drop(columns=["Label"]).select_dtypes(include=[np.number])

        X = X_df.iloc[:, :EXPECTED_FEATURES].values

        X_scaled = scaler.transform(X)

        X_seq = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        preds = model.predict(X_seq, batch_size=256, verbose=0)

        y_pred = np.argmax(preds, axis=1)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        acc = accuracy_score(y, y_pred)

        precision = precision_score(y, y_pred, average="weighted")

        recall = recall_score(y, y_pred, average="weighted")

        f1 = f1_score(y, y_pred, average="weighted")

        cm = confusion_matrix(y, y_pred).tolist()

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm
        }

    except Exception as e:

        import traceback

        return {"error": str(e), "trace": traceback.format_exc()}