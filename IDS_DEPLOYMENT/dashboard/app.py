import streamlit as st
import requests
import pandas as pd
import time

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Deep Learning IDS",
    layout="wide",
    page_icon="🚨"
)

st.title("🚨 Deep Learning IDS Dashboard")

# -----------------------------
# API ENDPOINTS
# -----------------------------
API_PREDICT = "http://127.0.0.1:8000/predict"
API_EVAL = "http://127.0.0.1:8000/evaluate"
API_SAMPLE = "http://127.0.0.1:8000/get_sample"

# -----------------------------
# ATTACK LABELS
# -----------------------------
attack_labels = {
    0: "Benign",
    1: "Bot",
    2: "DDoS",
    3: "DoS Hulk",
    4: "FTP-Patator",
    5: "PortScan",
    6: "Web Attack"
}

# -----------------------------
# API FUNCTIONS
# -----------------------------
def get_real_flow():
    try:
        response = requests.get(API_SAMPLE, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def predict_api(features, explain=True):
    try:
        response = requests.post(
            API_PREDICT,
            json={"features": features, "explain": explain},
            timeout=20
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# BUTTONS
# -----------------------------
col1, col2, col3 = st.columns(3)

generate = col1.button("🔍 Generate Network Flow", use_container_width=True)
start_live = col2.button("📡 Start Live Detection", use_container_width=True)
evaluate = col3.button("📊 Evaluate Model Accuracy", use_container_width=True)

# -----------------------------
# SINGLE ATTACK TEST
# -----------------------------
if generate:

    with st.spinner("Fetching real network flow and analyzing..."):

        sample = get_real_flow()

        if "error" in sample:
            st.error(sample["error"])

        else:

            features = sample["features"]
            actual_class = sample["actual_label"]

            actual_name = attack_labels.get(actual_class, "Unknown")

            result = predict_api(features, explain=True)

            if "prediction" in result:

                pred_class = result["prediction"]
                pred_name = attack_labels.get(pred_class, "Unknown")

                confidence = max(result["probabilities"]) * 100

                st.markdown("### Prediction Result")

                c1, c2, c3 = st.columns(3)

                c1.metric("Actual Label", actual_name)
                c2.metric("Model Prediction", pred_name)
                c3.metric("Confidence", f"{confidence:.2f}%")

                if pred_class == actual_class:
                    st.success(f"✅ Correctly classified as **{pred_name}**")
                else:
                    st.error(f"❌ Misclassified: predicted **{pred_name}** but actual is **{actual_name}**")

                # -----------------------------
                # PROBABILITY TABLE
                # -----------------------------
                st.write("### Prediction Probabilities")

                probs = result["probabilities"]

                prob_df = pd.DataFrame({
                    "Attack Type": [attack_labels[i] for i in range(len(probs))],
                    "Probability": probs
                })

                st.dataframe(prob_df)

                # -----------------------------
                # EXPLAINABLE AI
                # -----------------------------
                st.write("### Explainable AI (Top Features)")

                if result["top_features"]:

                    feature_df = pd.DataFrame({
                        "Feature": result["top_features"],
                        "Importance": result["importance_scores"]
                    })

                    st.bar_chart(feature_df.set_index("Feature"))

            else:
                st.error(result.get("error", "Unknown error"))

# -----------------------------
# LIVE TRAFFIC MONITORING
# -----------------------------
if start_live:

    st.markdown("### 📡 Live Traffic Monitoring")

    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    history = []

    for i in range(50):

        sample = get_real_flow()

        if "error" not in sample:

            features = sample["features"]

            result = predict_api(features, explain=False)

            if "prediction" in result:

                attack = result["prediction"]

                attack_name = attack_labels.get(attack, f"Class {attack}")

                if attack == 0:
                    status_placeholder.success(f"Current Status: {attack_name} ✅")
                else:
                    status_placeholder.error(f"Current Status: {attack_name} ⚠️")

            else:
                attack = -1
                status_placeholder.warning("Prediction Error")

        else:
            attack = -1

        history.append(attack)

        df = pd.DataFrame({"Prediction": history})

        chart_placeholder.line_chart(df)

        time.sleep(0.5)

# -----------------------------
# MODEL EVALUATION
# -----------------------------
if evaluate:

    with st.spinner("Evaluating model performance..."):

        try:

            response = requests.get(API_EVAL, timeout=30)

            result = response.json()

            if "accuracy" in result:

                st.markdown("### Model Performance Metrics")

                m1, m2, m3, m4 = st.columns(4)

                m1.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
                m2.metric("Precision", f"{result['precision']*100:.2f}%")
                m3.metric("Recall", f"{result['recall']*100:.2f}%")
                m4.metric("F1 Score", f"{result['f1_score']*100:.2f}%")

                # -----------------------------
                # CONFUSION MATRIX
                # -----------------------------
                if "confusion_matrix" in result:

                    st.write("### Confusion Matrix")

                    cm = pd.DataFrame(
                        result["confusion_matrix"],
                        columns=list(attack_labels.values()),
                        index=list(attack_labels.values())
                    )

                    st.dataframe(cm)

                    st.bar_chart(cm)

            else:

                st.error(result.get("error", "Evaluation failed"))

        except requests.exceptions.Timeout:
            st.error("Evaluation timeout. Try reducing dataset sample size.")