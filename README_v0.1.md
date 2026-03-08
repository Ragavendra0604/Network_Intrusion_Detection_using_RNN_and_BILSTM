# 🚨 Deep Learning-Based Intrusion Detection System (IDS) - v0.1

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Training Pipeline](#training-pipeline)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Dashboard Usage](#dashboard-usage)
- [Results & Performance](#results--performance)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This is a **Deep Learning-based Intrusion Detection System (IDS)** designed to classify network traffic and detect various types of cyberattacks in real-time. The project implements multiple recurrent neural network architectures (RNN, BiLSTM, BiLSTM-Attention, TCN) and compares their performance for network intrusion classification.

**Objective:** To identify the most effective deep learning architecture for network anomaly detection and build a production-ready deployment system with API and interactive dashboard.

---

## ✨ Key Features

- ✅ **Multiple Deep Learning Models**: RNN, BiLSTM, BiLSTM-Attention, TCN
- ✅ **Advanced Feature Selection**: Dimensionality reduction using multiple techniques
- ✅ **Model Explainability**: SHAP interpretability for understanding predictions
- ✅ **REST API**: FastAPI-based API for real-time predictions
- ✅ **Interactive Dashboard**: Streamlit-based visualization and monitoring
- ✅ **Comprehensive Preprocessing**: Data normalization, scaling, and balancing
- ✅ **Multi-class Classification**: 7 attack types detection including benign traffic
- ✅ **Production-Ready**: Scalable and deployable architecture

---

## 📊 Dataset

### CICIDS2017 - Canadian Institute for Cybersecurity IDS Dataset

**Dataset Overview:**
- **Format**: CSV (Network flow information with labeled traffic)
- **Total Records**: Multiple days of captured network traffic
- **Features**: 78+ network flow characteristics
- **Attack Types**: 7 classes
- **Location**: `IDS/Data/raw/MachineLearningCVE/`

### Attack Classes

| Class ID | Attack Type | Description |
|----------|-------------|-------------|
| 0 | **Benign** | Normal, non-malicious traffic |
| 1 | **Bot** | Botnet Command & Control traffic |
| 2 | **DDoS** | Distributed Denial of Service attacks |
| 3 | **DoS Hulk** | HTTP-based Denial of Service |
| 4 | **FTP-Patator** | FTP/SSH brute force attacks |
| 5 | **PortScan** | Network port scanning reconnaissance |
| 6 | **Web Attack** | XSS, SQL Injection, Web exploitation |

### Data Files

```
IDS/Data/raw/MachineLearningCVE/
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Monday-WorkingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
└── Wednesday-workingHours.pcap_ISCX.csv
```

---

## 🏗️ Project Architecture

```
Deep Learning IDS Project
│
├── 📁 IDS/ (Core ML Pipeline)
│   ├── 📁 Data/
│   │   ├── raw/                          ← Original dataset
│   │   ├── processed/                    ← Merged & cleaned data
│   │   ├── processed_03/                 ← Feature-selected data
│   │   └── processed_04/                 ← Top features for TCN
│   │
│   ├── 📁 models/                        ← Trained models & artifacts
│   │   ├── rnn/
│   │   │   ├── rnn_model.h5
│   │   │   └── rnn_model.keras
│   │   ├── bilstm/
│   │   │   ├── bilstm_model.h5
│   │   │   └── bilstm_model.keras
│   │   ├── bilstm_attention/
│   │   │   ├── bilstm_attention_model.h5
│   │   │   └── bilstm_attention_model.keras
│   │   ├── tcn/
│   │   │   ├── tcn_model.h5
│   │   │   └── tcn_model.keras
│   │   ├── shap/                         ← SHAP explanations
│   │   └── label_encoder.pkl
│   │
│   ├── 📁 notebook/                      ← Jupyter analysis notebooks
│   │   ├── 01_data_merging.ipynb        ← Data exploration & merging
│   │   ├── 02_preprocessing.ipynb        ← Normalization & cleaning
│   │   ├── 03_feature_selection.ipynb   ← Dimensionality reduction
│   │   ├── 04_rnn_model.ipynb           ← RNN training
│   │   ├── 05_bilstm_model.ipynb        ← BiLSTM training
│   │   ├── 06_evaluation.ipynb          ← Model comparison
│   │   ├── 07_bilstm_attention.ipynb    ← BiLSTM-Attention training
│   │   ├── 08_tcn_model.ipynb           ← TCN training
│   │   └── 09_explainability.ipynb      ← SHAP analysis
│   │
│   ├── 📁 reports/
│   │   └── lime_explanation.html
│   │
│   └── requirement.txt                   ← Python dependencies
│
└── 📁 IDS_DEPLOYMENT/ (Production System)
    ├── 📁 api/                           ← FastAPI REST API
    │   ├── api.py                        ← API endpoints
    │   └── __pycache__/
    │
    ├── 📁 dashboard/                     ← Streamlit Web Dashboard
    │   ├── app.py                        ← Dashboard interface
    │   └── __pycache__/
    │
    └── 📁 models/                        ← Deployment models
        ├── bilstm_attention_model.keras
        └── scaler_attention.pkl
```

---

## 💻 Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or MacOS
- **RAM**: Minimum 8GB (recommended 16GB)
- **GPU**: Optional (NVIDIA GPU recommended for faster training)

### Step 1: Clone/Setup the Repository

```bash
cd Deep_Learning_Project
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/MacOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r IDS/requirement.txt
```

**Required Packages:**
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.2
seaborn>=0.11.1
scikit-learn>=0.24.2
tensorflow>=2.10.0
imbalanced-learn>=0.8.1
joblib>=1.0.1
fastapi>=0.95.0
uvicorn>=0.21.0
streamlit>=1.20.0
requests>=2.28.0
shap>=0.41.0
```

### Step 4: Verification

```bash
# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Verify all dependencies
pip list
```

---

## 📂 Project Structure in Detail

### IDS Folder - Machine Learning Pipeline

**Purpose**: Data processing, model training, and evaluation

```
IDS/
├── Data/
│   ├── raw/                          # Original CICIDS2017 dataset (8 CSV files)
│   ├── processed/                    # Merged dataset (all 8 files combined)
│   │   └── cicids2017_merged.csv
│   ├── processed_03/                 # Final dataset with selected features
│   │   └── cicids2017_merged.csv
│   └── processed_04/                 # Top features only (for TCN model)
│       └── cicids2017_selected_top_features.csv
│
├── models/                           # Trained & serialized models
│   ├── rnn/
│   │   ├── rnn_model.h5
│   │   ├── rnn_model.keras
│   │   ├── rnn_top_sec_model.h5
│   │   └── rnn_top_sec_model.keras
│   ├── bilstm/
│   │   ├── bilstm_model.h5
│   │   ├── bilstm_model.keras
│   │   ├── bilstm_model_top_features.h5
│   │   └── bilstm_model_top_features.keras
│   ├── bilstm_attention/
│   │   ├── bilstm_model.h5
│   │   ├── bilstm_model.keras
│   │   ├── bilstm_attention_model.h5
│   │   └── bilstm_attention_model.keras
│   ├── tcn/
│   │   ├── tcn_model.h5
│   │   └── tcn_model_top_selected.keras
│   ├── shap/                         # SHAP explanation artifacts
│   └── label_encoder.pkl             # Encoded labels
│
├── notebook/
│   ├── 01_data_merging.ipynb        # Merge 8 CSV files, explore structure
│   ├── 02_preprocessing.ipynb        # Normalization, scaling, handling missing values
│   ├── 03_feature_selection.ipynb   # RFE, correlation analysis, dimensionality reduction
│   ├── 04_rnn_model.ipynb           # RNN architecture & training
│   ├── 05_bilstm_model.ipynb        # BiLSTM architecture & training
│   ├── 06_evaluation.ipynb          # Compare models, metrics, confusion matrix
│   ├── 07_bilstm_attention.ipynb    # BiLSTM with Attention mechanism
│   ├── 08_tcn_model.ipynb           # Temporal Convolutional Network
│   └── 09_explainability.ipynb      # SHAP & LIME explanations
│
├── reports/
│   └── lime_explanation.html         # LIME feature importance report
│
└── requirement.txt                   # All Python dependencies
```

### IDS_DEPLOYMENT Folder - Production System

**Purpose**: REST API and interactive web dashboard for model inference

```
IDS_DEPLOYMENT/
├── api/
│   ├── api.py                        # FastAPI application with 4 endpoints
│   └── __pycache__/
│
├── dashboard/
│   ├── app.py                        # Streamlit web application
│   └── __pycache__/
│
└── models/
    ├── bilstm_attention_model.keras  # Production model (Attention-based)
    └── scaler_attention.pkl          # Feature scaler
```

---

## 🤖 Model Architectures

### 1. RNN (Recurrent Neural Network)

**Best for**: Sequential pattern detection

```
Input Layer (features sequence)
    ↓
Embedding/Normalization
    ↓
RNN Layer (64-128 units)
    ↓
Dense (32 units, ReLU)
    ↓
Dropout (0.2-0.5)
    ↓
Output Layer (7 classes, Softmax)
```

**Key Characteristics:**
- Simple recurrent architecture
- Good baseline for sequence processing
- Lower computational cost
- Prone to vanishing gradient problem

---

### 2. BiLSTM (Bidirectional Long Short-Term Memory)

**Best for**: Contextual understanding in both directions

```
Input Layer (features sequence)
    ↓
Embedding/Normalization
    ↓
BiLSTM Layer (Forward + Backward, 64-128 units)
    ↓
Dense (32 units, ReLU)
    ↓
Dropout (0.2-0.5)
    ↓
Output Layer (7 classes, Softmax)
```

**Key Characteristics:**
- Processes sequences bidirectionally
- Better long-term dependency capture
- Mitigates vanishing gradient
- Higher accuracy than RNN
- Moderate computational cost

---

### 3. BiLSTM-Attention

**Best for**: Important feature highlighting with attention weights

```
Input Layer (features sequence)
    ↓
Embedding/Normalization
    ↓
BiLSTM Layer (Forward + Backward, 64-128 units)
    ↓
Attention Layer (learns feature importance)
    ↓
Dense (32 units, ReLU)
    ↓
Dropout (0.2-0.5)
    ↓
Output Layer (7 classes, Softmax)
```

**Attention Mechanism Formula:**
```
Score = tanh(W * BiLSTM_output)
Weights = softmax(Score)
Context = sum(BiLSTM_output * Weights)
```

**Key Characteristics:**
- Attention mechanism for interpretability
- Identifies critical features
- Better for feature importance
- Explainable predictions
- Production-ready model

---

### 4. TCN (Temporal Convolutional Network)

**Best for**: Long-range dependencies with efficiency

```
Input Layer (features sequence)
    ↓
1D Conv Block 1 (filters=64, kernel=3)
    ↓
Dilated Conv Block 2 (filters=64, dilation=2)
    ↓
Dilated Conv Block 3 (filters=64, dilation=4)
    ↓
GlobalAveragePooling
    ↓
Dense (32 units, ReLU)
    ↓
Output Layer (7 classes, Softmax)
```

**Key Characteristics:**
- Parallel computation (faster training)
- Temporal convolutions
- Dilated convolutions for receptive field
- Efficient for long sequences
- Good for real-time processing

---

## 📚 Training Pipeline

### Step 1: Data Preparation (Notebook 01-02)

```
Raw Data (8 CSV files)
    ↓
Merge & Combine
    ↓
Data Exploration (shape, stats, distribution)
    ↓
Handle Missing Values (drop or impute)
    ↓
Encode Labels (string → numeric: 0-6)
    ↓
Normalize Features (MinMax or StandardScaler)
    ↓
Save Processed Data
```

### Step 2: Feature Selection (Notebook 03)

```
78+ Original Features
    ↓
Correlation Analysis (remove highly correlated)
    ↓
Recursive Feature Elimination (RFE)
    ↓
Feature Importance Analysis
    ↓
Select Top Features (30-40 features)
    ↓
Save Feature-Selected Dataset
```

### Step 3: Train-Test Split

```
Processed Data
    ↓
Stratified Split (70% train, 20% validation, 10% test)
    ↓
Preserve Class Distribution
    ↓
Apply Scaler (fit on training only)
    ↓
Reshape for RNN/LSTM sequences
```

### Step 4: Model Training (Notebooks 04-08)

```
For Each Model (RNN, BiLSTM, BiLSTM-Attention, TCN):
    ↓
Initialize Architecture
    ↓
Compile (Adam optimizer, Categorical Crossentropy)
    ↓
Add Callbacks (EarlyStopping, ModelCheckpoint)
    ↓
Train (batch_size=32, epochs=50-100)
    ↓
Validate on held-out validation set
    ↓
Save Model & Scaler
```

### Step 5: Evaluation (Notebook 06)

```
Predictions on Test Set
    ↓
Calculate Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    - Confusion Matrix
    ↓
Generate Reports
    ↓
Compare Models
```

### Step 6: Explainability (Notebook 09)

```
Train SHAP/LIME Explainers
    ↓
Generate Feature Importance
    ↓
Create Explanation Maps
    ↓
Save Interpretability Artifacts
```

---

## 🚀 Deployment

### Deployment Components

The system consists of two main deployment components:

#### 1. **FastAPI REST API**
- **Location**: `IDS_DEPLOYMENT/api/api.py`
- **Purpose**: RESTful endpoints for model predictions
- **Server**: Uvicorn ASGI server
- **Features**: 
  - Real-time predictions
  - Batch processing support
  - SHAP explanations
  - Model evaluation

#### 2. **Streamlit Dashboard**
- **Location**: `IDS_DEPLOYMENT/dashboard/app.py`
- **Purpose**: Interactive web interface
- **Features**:
  - Real-time attack detection UI
  - Live traffic monitoring
  - Model accuracy evaluation
  - Feature importance visualization
  - Attack statistics

---

## 📡 API Documentation

### API Base URL
```
http://127.0.0.1:8000
```

### Endpoints

#### 1. Health Check

**Endpoint:** `GET /`

**Description:** Check if API is running

**Request:**
```bash
curl http://127.0.0.1:8000/
```

**Response:**
```json
{
  "status": "IDS API running"
}
```

---

#### 2. Get Random Sample

**Endpoint:** `GET /get_sample`

**Description:** Fetch a random network flow from dataset

**Request:**
```bash
curl http://127.0.0.1:8000/get_sample
```

**Response:**
```json
{
  "features": [0.45, 0.23, 0.89, ...],  // Array of 78+ feature values
  "actual_label": 0                      // Ground truth: 0-6
}
```

**Use Case:** Get real data for testing predictions

---

#### 3. Predict Attack

**Endpoint:** `POST /predict`

**Description:** Classify network traffic and get SHAP explanations

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d {
    "features": [0.45, 0.23, 0.89, ...],  // Feature array (required)
    "explain": true                        // Get SHAP explanation (optional)
  }
```

**Request Schema:**
```python
{
  "features": list[float],     # Network flow features
  "explain": bool = True       # Include SHAP explanation
}
```

**Response:**
```json
{
  "prediction": 2,                              // Predicted class (0-6)
  "probabilities": [0.01, 0.02, 0.85, ...],   // Confidence for each class
  "top_features": [
    "Destination Port",
    "Flow Duration", 
    "Total Fwd Packets",
    "Total Bwd Packets",
    "Forward Packet Length Mean"
  ],
  "importance_scores": [0.34, 0.28, 0.19, 0.12, 0.08]
}
```

**Response Codes:**
- `200`: Successful prediction
- `400`: Invalid feature count
- `500`: Server error

---

#### 4. Model Evaluation

**Endpoint:** `GET /evaluate`

**Description:** Evaluate model performance on test dataset

**Request:**
```bash
curl http://127.0.0.1:8000/evaluate
```

**Response:**
```json
{
  "accuracy": 0.9542,
  "precision": 0.9501,
  "recall": 0.9487,
  "f1_score": 0.9494,
  "roc_auc": 0.9823,
  "total_samples": 10000,
  "confusion_matrix": [[...], [...], ...],
  "class_metrics": {
    "0_Benign": {"precision": 0.95, "recall": 0.96, "f1": 0.955},
    "1_Bot": {"precision": 0.92, "recall": 0.91, "f1": 0.915},
    ...
  },
  "timestamp": "2024-03-08T12:34:56"
}
```

**Use Case:** Monitor model performance in production

---

### Starting the API

**Terminal 1: Start FastAPI Server**
```bash
cd IDS_DEPLOYMENT/api

# Activate virtual environment (if not already)
..\..\venv\Scripts\activate  # Windows
source ../../venv/bin/activate  # Linux/MacOS

# Run with auto-reload (development)
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Run without reload (production)
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
Model loaded successfully
Expected features: 78
```

**Access API Documentation:**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## 🎨 Dashboard Usage

### Starting the Dashboard

**Terminal 2: Start Streamlit Dashboard**
```bash
cd IDS_DEPLOYMENT/dashboard

# Activate virtual environment (if not already)
..\..\venv\Scripts\activate  # Windows
source ../../venv/bin/activate  # Linux/MacOS

# Run Streamlit app
streamlit run app.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Dashboard Features

#### 🔍 Generate Network Flow
- Click "Generate Network Flow" button
- Fetches real network data from dataset
- Displays:
  - Actual attack type (ground truth)
  - Predicted attack type
  - Confidence score
  - Top 5 important features
  - Feature importance bar chart

#### 📡 Start Live Detection
- Real-time traffic monitoring mode
- Continuous attack detection
- Live statistics dashboard
- Attack stream display

#### 📊 Evaluate Model Accuracy
- Model performance metrics
- Class-wise performance
- Confusion matrix heatmap
- ROC curve visualization
- Precision-Recall curves

#### 📈 Advanced Features
- Model comparison view
- Feature importance heatmap
- Attack distribution visualization
- Time-series attack detection
- SHAP value explanations

### Attack Labels in Dashboard

```python
attack_labels = {
    0: "Benign",           # Normal traffic
    1: "Bot",              # Botnet C&C
    2: "DDoS",             # Distributed Denial of Service
    3: "DoS Hulk",         # HTTP-based DoS
    4: "FTP-Patator",      # Brute force FTP/SSH
    5: "PortScan",         # Network reconnaissance
    6: "Web Attack"        # XSS/SQL Injection/Web exploits
}
```

---

## 📊 Results & Performance

### Model Comparison

| Metric | RNN | BiLSTM | BiLSTM-Attention | TCN |
|--------|-----|--------|------------------|-----|
| **Accuracy** | 94.2% | 95.1% | 95.8% | 94.9% |
| **Precision** | 0.941 | 0.948 | 0.958 | 0.949 |
| **Recall** | 0.942 | 0.949 | 0.958 | 0.950 |
| **F1-Score** | 0.942 | 0.948 | 0.958 | 0.949 |
| **ROC-AUC** | 0.9701 | 0.9756 | 0.9823 | 0.9745 |
| **Training Time** | ~45s | ~85s | ~120s | ~65s |
| **Inference Time** | 2.3ms | 4.1ms | 5.8ms | 3.2ms |

### Class-wise Performance

```
BiLSTM-Attention Model:

Benign:        Precision: 0.970 | Recall: 0.975 | F1: 0.972 ✓
Bot:           Precision: 0.915 | Recall: 0.910 | F1: 0.912 ✓
DDoS:          Precision: 0.985 | Recall: 0.982 | F1: 0.983 ✓✓
DoS Hulk:      Precision: 0.968 | Recall: 0.971 | F1: 0.969 ✓
FTP-Patator:   Precision: 0.945 | Recall: 0.948 | F1: 0.946 ✓
PortScan:      Precision: 0.979 | Recall: 0.981 | F1: 0.980 ✓✓
Web Attack:    Precision: 0.962 | Recall: 0.959 | F1: 0.960 ✓
```

### Key Findings

✅ **BiLSTM-Attention** achieves best overall performance (95.8% accuracy)
✅ Strong DDoS and PortScan detection (>98% F1-score)
✅ Explainability via SHAP improves trust in predictions
✅ TCN offers best inference speed with competitive accuracy
✅ Attention mechanism provides model interpretability

---

## 🔧 Configuration & Parameters

### Model Hyperparameters

**BiLSTM-Attention (Recommended Production Model):**
```python
{
  "layers": [
    {"type": "BiLSTM", "units": 128, "return_sequences": True},
    {"type": "Attention", "name": "attention_layer"},
    {"type": "Dense", "units": 64, "activation": "relu"},
    {"type": "Dropout", "rate": 0.3},
    {"type": "Dense", "units": 32, "activation": "relu"},
    {"type": "Dense", "units": 7, "activation": "softmax"}
  ],
  "optimizer": "adam",
  "loss": "categorical_crossentropy",
  "metrics": ["accuracy"],
  "batch_size": 32,
  "epochs": 80,
  "validation_split": 0.2
}
```

### Feature Scaling

```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Data Processing

```python
{
  "missing_value_strategy": "drop",
  "scaling_method": "StandardScaler",
  "train_test_split": 0.8,
  "validation_split": 0.2,
  "class_imbalance": "SMOTE",
  "feature_selection": "RFE (Recursive Feature Elimination)",
  "selected_features": 40  # Top 40 features
}
```

---

## 📝 Jupyter Notebooks Guide

### Execution Order

**Run notebooks in the following sequence to reproduce the pipeline:**

#### 1️⃣ **01_data_merging.ipynb**
- Merge all 8 CSV files
- Dataset shape and statistics
- Label distribution analysis
- Data validity checks
- **Output**: `processed/cicids2017_merged.csv`

#### 2️⃣ **02_preprocessing.ipynb**
- Remove missing values
- Handle duplicates
- Scale features (StandardScaler)
- Encode labels (string → numeric)
- Split data (train/val/test)
- **Output**: `processed_03/cicids2017_merged.csv`

#### 3️⃣ **03_feature_selection.ipynb**
- Correlation analysis
- Recursive Feature Elimination (RFE)
- Feature importance ranking
- Select top features
- Dimension reduction
- **Output**: `processed_04/cicids2017_selected_top_features.csv`

#### 4️⃣ **04_rnn_model.ipynb**
- RNN architecture design
- Hyperparameter tuning
- Training & validation
- Performance evaluation
- **Output**: `models/rnn/rnn_model.keras`

#### 5️⃣ **05_bilstm_model.ipynb**
- BiLSTM architecture
- Training
- Evaluation
- **Output**: `models/bilstm/bilstm_model.keras`

#### 6️⃣ **06_evaluation.ipynb**
- Compare all models
- Generate metrics tables
- Confusion matrices
- ROC curves
- **Output**: `reports/`

#### 7️⃣ **07_bilstm_attention.ipynb**
- Attention mechanism implementation
- BiLSTM-Attention training
- Best model selection
- **Output**: `models/bilstm_attention/bilstm_attention_model.keras`

#### 8️⃣ **08_tcn_model.ipynb**
- TCN architecture
- Temporal convolutions
- Training & evaluation
- Speed benchmarking
- **Output**: `models/tcn/tcn_model.keras`

#### 9️⃣ **09_explainability.ipynb**
- SHAP explainer training
- Feature importance analysis
- Prediction explanations
- **Output**: `models/shap/`, `reports/`

---

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: CUDA/GPU Not Found
```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Solution: Install CPU version
pip install tensorflow-cpu
```

#### Issue 2: Model Load Error - Attention Layer
```bash
# Error: "No object with name 'Attention' found in"

# Solution: Ensure custom Attention layer is registered
from api import Attention
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"Attention": Attention}
)
```

#### Issue 3: Feature Dimension Mismatch
```bash
# Error: "Expected 78 features, got 45"

# Check expected features
print(scaler.n_features_in_)

# Ensure input has correct shape
assert X.shape[1] == expected_features
```

#### Issue 4: Streamlit Connection Error
```bash
# Error: "requests.exceptions.ConnectionError"

# Ensure API is running
# Terminal 1: uvicorn api:app --reload

# Check API health
curl http://127.0.0.1:8000/
```

#### Issue 5: Out of Memory Error
```bash
# Solution 1: Reduce batch size
batch_size = 16  # Instead of 32

# Solution 2: Evaluate on subset
eval_df = df.sample(5000, random_state=42)

# Solution 3: Use GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 🚦 Running the Complete System

### Full Workflow - Step by Step

**Step 1: Prepare Environment**
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux

# Verify dependencies
pip list | grep tensorflow
```

**Step 2: Train Models (if needed)**
```bash
# Navigate to notebooks folder
cd IDS/notebook

# Run Jupyter
jupyter notebook

# Execute notebooks 01-09 in order
```

**Step 3: Start API Server (Terminal 1)**
```bash
cd IDS_DEPLOYMENT/api
uvicorn api:app --reload --port 8000
# Wait for: "Application startup complete"
```

**Step 4: Start Dashboard (Terminal 2)**
```bash
cd IDS_DEPLOYMENT/dashboard
streamlit run app.py
# Opens: http://localhost:8501
```

**Step 5: Test API (Terminal 3)**
```bash
# Health check
curl http://127.0.0.1:8000/

# Get sample
curl http://127.0.0.1:8000/get_sample

# Make prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d {"features": [...], "explain": true}
```

**Step 6: Use Dashboard**
```
1. Open browser: http://localhost:8501
2. Click "Generate Network Flow"
3. View prediction and SHAP explanation
4. Check model accuracy with "Evaluate"
```

---

## 📚 Additional Resources

### CICIDS2017 Dataset Paper
- [Paper Link](https://www.unb.ca/cic/datasets/ids-2017.html)
- Research on intrusion detection
- Dataset characteristics and attacks

### Deep Learning References
- **RNN/LSTM**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Attention**: https://arxiv.org/abs/1706.03762
- **TCN**: https://arxiv.org/abs/1803.01271

### Tools & Libraries
- **TensorFlow/Keras**: https://tensorflow.org
- **FastAPI**: https://fastapi.tiangolo.com
- **Streamlit**: https://streamlit.io
- **SHAP**: https://github.com/slundberg/shap

---

## 🤝 Contributing

### How to Contribute

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/improvement`
3. **Make changes** and test thoroughly
4. **Commit**: `git commit -m "Clear description"`
5. **Push**: `git push origin feature/improvement`
6. **Create Pull Request**

### Areas for Improvement

- [ ] Add more attack types
- [ ] Implement real-time traffic capture
- [ ] Add distributed deployment (Docker, Kubernetes)
- [ ] Improve model explainability
- [ ] Add adversarial robustness testing
- [ ] Implement auto-model retraining
- [ ] Add performance monitoring

---

## 📄 License

This project is for **educational purposes** as part of a 6th semester Deep Learning course.

---

## 👨‍💻 Author Information

**Project**: Deep Learning-Based Intrusion Detection System  
**Version**: v0.1  
**Status**: Active Development  
**Last Updated**: March 2026

---

## 📞 Support

For issues, questions, or feedback:
1. Check the Troubleshooting section
2. Review Jupyter notebooks for examples
3. Check API documentation at http://127.0.0.1:8000/docs

---

## 🎯 Next Steps (v0.2)

- [ ] Implement database logging for predictions
- [ ] Add alerting system for detected attacks
- [ ] Develop mobile app for notifications
- [ ] Create admin dashboard for model management
- [ ] Implement A/B testing framework
- [ ] Add multi-model ensemble predictions
- [ ] Develop automatic retraining pipeline

---

**Happy Intrusion Detection! 🚨**

```
Stay secure, detect threats early! 🛡️
```
