"""
generate_dataset.py
-------------------
Generates a synthetic dataset for calorie burn prediction and saves it as calories_dataset.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

age        = np.random.randint(18, 70, N)
gender     = np.random.choice([0, 1], N)          # 0 = Female, 1 = Male
weight_kg  = np.round(np.random.uniform(45, 120, N), 1)
height_cm  = np.round(np.random.uniform(150, 200, N), 1)
duration   = np.random.randint(10, 90, N)          # minutes
heart_rate = np.random.randint(80, 185, N)
body_temp  = np.round(np.random.uniform(36.0, 41.0, N), 1)
exercise   = np.random.choice(
    ['Running', 'Cycling', 'Swimming', 'Walking', 'HIIT', 'Yoga', 'Weightlifting'],
    N
)

exercise_factor = {
    'Running': 1.15, 'Cycling': 1.05, 'Swimming': 1.20,
    'Walking': 0.70, 'HIIT': 1.30, 'Yoga': 0.55, 'Weightlifting': 0.90
}
ef = np.array([exercise_factor[e] for e in exercise])

bmi = weight_kg / (height_cm / 100) ** 2

# Physiologically inspired formula
calories = (
    (0.55 * heart_rate)
    + (0.36 * weight_kg)
    - (0.20 * age)
    + (2.0  * duration)
    + (1.8  * body_temp)
    + (8.0  * gender)          # males burn slightly more at rest
) * ef + np.random.normal(0, 12, N)

calories = np.round(np.clip(calories, 20, None), 2)

df = pd.DataFrame({
    'Age':           age,
    'Gender':        np.where(gender == 1, 'Male', 'Female'),
    'Weight_kg':     weight_kg,
    'Height_cm':     height_cm,
    'BMI':           np.round(bmi, 2),
    'Duration_min':  duration,
    'Heart_Rate':    heart_rate,
    'Body_Temp_C':   body_temp,
    'Exercise_Type': exercise,
    'Calories_Burned': calories
})

df.to_csv('calories_dataset.csv', index=False)
print(f"Dataset saved: calories_dataset.csv  ({N} rows)")
print(df.describe())
