# Crop Recommendation System - Setup Instructions

## Files Created ✅
- [x] `train.py` - Model training script
- [x] `app.py` - Flask web application with white theme
- [x] `requirements.txt` - Python dependencies

## Setup Steps (To be executed)

### 1. Install Dependencies
```
bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional - model already exists)
```
bash
python train.py
```
This will train a new model and save it to `crop_model.pkl`

### 3. Run the Web Application
```
bash
python app.py
```

### 4. Open in Browser
Navigate to: `http://127.0.0.1:5000`

## Features Included

### White Theme UI
- Clean white background (#ffffff)
- Dark text (#1a1a1a)
- Green accent colors for headers (#1a472a)
- Professional card-based layout

### Input Fields
- Nitrogen (N): 0-140
- Phosphorus (P): 0-145
- Potassium (K): 0-205
- Temperature: 0-50°C
- Humidity: 0-100%
- pH Level: 3-9
- Rainfall: 0-300mm

### Output
- Top 5 crop recommendations with probabilities
- Recommended crop highlighted
- Farming tips section

## Project Structure
```
Crop-Advisory-Sysytem/
├── app.py              # Flask web application
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── crop_model.pkl      # Trained model (already exists)
├── Crop_recommendation.csv  # Dataset
└── README.md           # Project documentation
