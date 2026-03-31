# 🔥 CalorieIQ — AI Calorie Burn Predictor

> A complete end-to-end machine learning project that predicts calories burned during exercise, built with scikit-learn, a synthetic dataset generator, and an interactive dark-themed frontend.

---

## 📁 Project Structure

```
calorie-burn-predictor/
│
├── generate_dataset.py      # Generates synthetic training data (calories_dataset.csv)
├── train_model.py           # Trains ML models and saves the best one (calorie_model.pkl)
├── index.html               # Interactive frontend for real-time predictions
│
├── calories_dataset.csv     # Auto-generated after running generate_dataset.py
├── calorie_model.pkl        # Auto-generated after running train_model.py
│
├── feature_importance.png   # Auto-generated plot
├── actual_vs_predicted.png  # Auto-generated plot
└── residuals.png            # Auto-generated plot
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Generate the dataset

```bash
python generate_dataset.py
```

Creates `calories_dataset.csv` with **2,000 synthetic rows**.

### 3. Train the model

```bash
python train_model.py
```

Trains 4 models, selects the best by R², saves it as `calorie_model.pkl`, and exports 3 diagnostic plots.

### 4. Open the frontend

Simply open `index.html` in any modern browser — no server required.

---

## 📊 Dataset

The dataset is generated synthetically using a physiologically-inspired formula.

| Feature | Description |
|---|---|
| `Age` | 18 – 70 years |
| `Gender` | Male / Female |
| `Weight_kg` | 45 – 120 kg |
| `Height_cm` | 150 – 200 cm |
| `BMI` | Derived from weight and height |
| `Duration_min` | 10 – 90 minutes |
| `Heart_Rate` | 80 – 185 bpm |
| `Body_Temp_C` | 36.0 – 41.0 °C |
| `Exercise_Type` | Running, Cycling, Swimming, Walking, HIIT, Yoga, Weightlifting |
| `Calories_Burned` | **Target variable** (kcal) |

Two engineered features are added during training: `HR × Duration` and `Weight × Duration`.

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2-regularised baseline |
| Random Forest | 200 estimators, parallel |
| **Gradient Boosting** | **Best performer (default winner)** |

The best model is selected automatically by test-set R² and saved as `calorie_model.pkl`.

---

## 📈 Output Plots

After training, three diagnostic plots are saved:

- **`feature_importance.png`** — Top 12 most influential features
- **`actual_vs_predicted.png`** — Scatter plot of ground truth vs model output
- **`residuals.png`** — Distribution of prediction errors

---

## 🖥️ Frontend

`index.html` is a fully self-contained, zero-dependency dark-themed web app.

**Controls:**
- Age, Weight, Height inputs
- Gender toggle (Male / Female)
- Sliders for Duration, Heart Rate, Body Temperature
- Exercise type selection pills (7 types)

**Result panel shows:**
- Estimated calories burned (animated counter)
- Burn intensity progress bar
- BMI, kcal/min, and pizza-slice equivalent
- Contextual workout tip per exercise type

> The frontend uses the same formula as the training script, so predictions are consistent without needing a backend server.

---

## 🔬 Quick Inference (Python)

After training, you can load the model and predict directly:

```python
import joblib, pandas as pd

model = joblib.load('calorie_model.pkl')

sample = pd.DataFrame([{
    'Age': 28, 'Gender': 'Male', 'Weight_kg': 75.0, 'Height_cm': 178.0,
    'BMI': 23.7, 'Duration_min': 45, 'Heart_Rate': 148, 'Body_Temp_C': 38.2,
    'Exercise_Type': 'Running',
    'HR_Duration': 148 * 45, 'Weight_Duration': 75.0 * 45
}])

calories = model.predict(sample)[0]
print(f"Predicted: {calories:.1f} kcal")
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data generation | Python, NumPy, Pandas |
| ML pipeline | scikit-learn (Pipeline, ColumnTransformer) |
| Visualisation | Matplotlib, Seaborn |
| Model persistence | joblib |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Fonts | Syne + DM Sans (Google Fonts) |

---

## 📝 Notes

- The dataset is **synthetic** and intended for educational/demo purposes.
- Calorie predictions are estimates based on common exercise physiology approximations.
- Swap in a real-world dataset (e.g. from Fitbit or a wearable API) and re-run `train_model.py` to get production-grade accuracy.

---

## 📄 License

MIT — free to use, modify, and distribute.
