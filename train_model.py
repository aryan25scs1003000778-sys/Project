"""
train_model.py
--------------
Trains multiple regression models on calories_dataset.csv,
evaluates them, picks the best, and saves it as calorie_model.pkl.

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import OneHotEncoder
from sklearn.linear_model      import LinearRegression, Ridge
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score


# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  CALORIE BURN PREDICTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv('calories_dataset.csv')
print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(df.head(3).to_string())

# ── 2. Feature Engineering ────────────────────────────────────────────────────
df['HR_Duration']     = df['Heart_Rate'] * df['Duration_min']
df['Weight_Duration'] = df['Weight_kg']  * df['Duration_min']

X = df.drop(columns=['Calories_Burned'])
y = df['Calories_Burned']

num_features = ['Age', 'Weight_kg', 'Height_cm', 'BMI',
                'Duration_min', 'Heart_Rate', 'Body_Temp_C',
                'HR_Duration', 'Weight_Duration']
cat_features = ['Gender', 'Exercise_Type']

# ── 3. Preprocessing Pipeline ────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
])

# ── 4. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[INFO] Train size: {len(X_train)}   Test size: {len(X_test)}")

# ── 5. Define Models ──────────────────────────────────────────────────────────
models = {
    'Linear Regression':       LinearRegression(),
    'Ridge Regression':        Ridge(alpha=1.0),
    'Random Forest':           RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':       GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                          max_depth=5, random_state=42),
}

results = {}
trained_pipelines = {}

# ── 6. Train & Evaluate ───────────────────────────────────────────────────────
print("\n[INFO] Training models …\n")
for name, model in models.items():
    pipe = Pipeline([('pre', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    cv_r2  = cross_val_score(pipe, X_train, y_train, cv=5,
                              scoring='r2', n_jobs=-1).mean()

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'CV R²': cv_r2}
    trained_pipelines[name] = pipe

    print(f"  {name:<25}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:.4f}  CV R²={cv_r2:.4f}")

# ── 7. Select Best Model ──────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['R²'])
best_pipe  = trained_pipelines[best_name]
print(f"\n[BEST] {best_name}  (R² = {results[best_name]['R²']:.4f})")

# ── 8. Save Model ─────────────────────────────────────────────────────────────
joblib.dump(best_pipe, 'calorie_model.pkl')
print("[SAVED] calorie_model.pkl")

# ── 9. Feature Importance (if tree-based) ────────────────────────────────────
if hasattr(best_pipe.named_steps['model'], 'feature_importances_'):
    ohe_cols = list(
        best_pipe.named_steps['pre']
        .named_transformers_['cat']
        .get_feature_names_out(cat_features)
    )
    all_features = num_features + ohe_cols
    importances  = best_pipe.named_steps['model'].feature_importances_

    fi_df = (pd.DataFrame({'Feature': all_features, 'Importance': importances})
             .sort_values('Importance', ascending=False)
             .head(12))

    plt.figure(figsize=(9, 5))
    sns.barplot(data=fi_df, x='Importance', y='Feature',
                palette='YlOrRd_r', edgecolor='black', linewidth=0.6)
    plt.title(f'Feature Importances — {best_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=140)
    plt.close()
    print("[SAVED] feature_importance.png")

# ── 10. Actual vs Predicted Plot ─────────────────────────────────────────────
y_pred_best = best_pipe.predict(X_test)
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_best, alpha=0.45, s=18,
            color='#E8603C', edgecolors='none')
lim = [y_test.min() - 10, y_test.max() + 10]
plt.plot(lim, lim, 'k--', linewidth=1.2, label='Perfect fit')
plt.xlabel('Actual Calories', fontsize=12)
plt.ylabel('Predicted Calories', fontsize=12)
plt.title(f'Actual vs Predicted — {best_name}', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=140)
plt.close()
print("[SAVED] actual_vs_predicted.png")

# ── 11. Residual Distribution ─────────────────────────────────────────────────
residuals = y_test - y_pred_best
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=50, kde=True, color='#3A7BD5', edgecolor='white')
plt.axvline(0, color='red', linestyle='--', linewidth=1.2)
plt.xlabel('Residual (Actual − Predicted)', fontsize=12)
plt.title('Residual Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('residuals.png', dpi=140)
plt.close()
print("[SAVED] residuals.png")

# ── 12. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
summary = pd.DataFrame(results).T.sort_values('R²', ascending=False)
print(summary.to_string(float_format='{:.4f}'.format))
print("\nAll artefacts saved. Training complete ✓")


# ── 13. Quick Inference Demo ──────────────────────────────────────────────────
print("\n[DEMO] Single prediction example:")
sample = pd.DataFrame([{
    'Age': 28, 'Gender': 'Male', 'Weight_kg': 75.0, 'Height_cm': 178.0,
    'BMI': 23.7, 'Duration_min': 45, 'Heart_Rate': 148, 'Body_Temp_C': 38.2,
    'Exercise_Type': 'Running',
    'HR_Duration': 148 * 45, 'Weight_Duration': 75.0 * 45
}])
pred = best_pipe.predict(sample)[0]
print(f"  Predicted calories burned: {pred:.1f} kcal")
