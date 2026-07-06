# Student Academic Performance Predictor

An end-to-end Machine Learning project that leverages Supervised Learning to predict a student's final marks. The model implements feature scaling and evaluates key academic indicators to provide continuous performance predictions.

---

## Project Overview
This repository contains a production-ready machine learning pipeline built using Python and Scikit-Learn. The system handles custom-curated tabular data containing student metrics to fit a Linear Regression model.

### Key Performance Indicators (Features) Used:
* **Hours Studied**: Total continuous study hours.
* **Attendance**: Percentage of total lectures attended.
* **Previous Marks**: Historical performance baseline out of 100.
* **Final Marks**: The continuous target variable to be predicted.

---

## Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (LinearRegression, StandardScaler, train_test_split)
* **Data Visualization:** Matplotlib

---

## 📈 Model Performance & Evaluation Metrics
The model was trained using an 80-20 train-test split configuration. After fitting the multi-variable data on a StandardScaler pipeline, the model achieved the following performance metrics on unseen testing data:

* **Mean Absolute Error (MAE):** 0.61 (Indicating near-perfect alignment with actual values)
* **Mean Squared Error (MSE):** 0.55
* **Root Mean Squared Error (RMSE):** 0.74
* **R-squared ($R^2$) Score:** 0.998 (Capturing 99.8% of the data variance)

---

## How It Works (Pipeline Steps)
1. **Data Preprocessing:** Handled custom tabular input data structures.
2. **Feature Scaling:** Applied `StandardScaler` to uniform the ranges of input variables preventing mathematical feature bias.
3. **Model Fitting:** Trained a multivariate `LinearRegression` algorithm.
4. **Validation & Visualization:** Plotted target distributions and validated performance using error residue metrics.
