"""
Crop Recommendation Model Training Script
=========================================
This script trains a RandomForestClassifier on the crop recommendation dataset
and saves the trained model and label encoder for later use.

Features: N, P, K, temperature, humidity, pH, rainfall
Target: Crop label
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100

# ============================================================================
# Load and Prepare Data
# ============================================================================
print("=" * 60)
print("CROP RECOMMENDATION MODEL TRAINING")
print("=" * 60)

print("\n[1] Loading dataset...")
data = pd.read_csv(DATA_PATH)
print(f"    Dataset loaded successfully!")
print(f"    Shape: {data.shape}")
print(f"    Columns: {list(data.columns)}")

print("\n[2] Exploring data...")
print(f"    Number of unique crops: {data['label'].nunique()}")
print(f"    Crops: {', '.join(data['label'].unique())}")

print("\n[3] Checking for missing values...")
missing = data.isnull().sum()
if missing.sum() == 0:
    print("    No missing values found!")
else:
    print(f"    Missing values:\n{missing[missing > 0]}")

print("\n[4] Data Statistics:")
print(data.describe().to_string())

# ============================================================================
# Prepare Features and Target
# ============================================================================
print("\n[5] Preparing features and target...")

# Features (X) - all columns except 'label'
X = data.drop('label', axis=1)

# Target (y) - the crop label
y = data['label']

print(f"    Features shape: {X.shape}")
print(f"    Target shape: {y.shape}")
print(f"    Feature names: {list(X.columns)}")

# ============================================================================
# Encode Target Labels
# ============================================================================
print("\n[6] Encoding target labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"    Number of classes: {len(le.classes_)}")
print(f"    Classes: {list(le.classes_)}")

# ============================================================================
# Train-Test Split
# ============================================================================
print("\n[7] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE
)

print(f"    Training set size: {X_train.shape[0]}")
print(f"    Test set size: {X_test.shape[0]}")

# ============================================================================
# Train Model
# ============================================================================
print("\n[8] Training Random Forest Classifier...")
print(f"    Number of estimators: {N_ESTIMATORS}")
print(f"    Random state: {RANDOM_STATE}")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1  # Use all available cores
)

model.fit(X_train, y_train)
print("    Model training completed!")

# ============================================================================
# Evaluate Model
# ============================================================================
print("\n[9] Evaluating model...")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"    Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\n[10] Classification Report:")
print(classification_report(
    y_test, y_pred, 
    target_names=le.classes_,
    zero_division=0
))

# ============================================================================
# Feature Importance
# ============================================================================
print("\n[11] Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# Save Model and Label Encoder
# ============================================================================
print("\n[12] Saving model and label encoder...")

joblib.dump(model, MODEL_PATH)
print(f"    Model saved to: {MODEL_PATH}")

joblib.dump(le, ENCODER_PATH)
print(f"    Label encoder saved to: {ENCODER_PATH}")

# ============================================================================
# Verify Saved Models
# ============================================================================
print("\n[13] Verifying saved models...")

# Load and test
loaded_model = joblib.load(MODEL_PATH)
loaded_encoder = joblib.load(ENCODER_PATH)

# Quick prediction test
sample_input = X_test.iloc[0:1]
prediction = loaded_model.predict(sample_input)
predicted_crop = loaded_encoder.inverse_transform(prediction)[0]

print(f"    Sample prediction: {predicted_crop}")
print(f"    Model verification: SUCCESS")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nSummary:")
print(f"  - Dataset: {DATA_PATH}")
print(f"  - Total samples: {len(data)}")
print(f"  - Number of crops: {len(le.classes_)}")
print(f"  - Model accuracy: {accuracy * 100:.2f}%")
print(f"  - Model saved: {MODEL_PATH}")
print(f"  - Encoder saved: {ENCODER_PATH}")
print("\n" + "=" * 60)
