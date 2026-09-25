# End to End MLOps Pipeline

Predictive Maintenance: End-to-End MLOps Pipeline

A full MLOps pipeline for multi-class failure prediction on industrial IoT sensor data, covering schema validation, experiment tracking, hyperparameter tuning, drift detection, and model explainability.

Business Context

Simulates a manufacturing environment where machines stream sensor data continuously and unplanned downtime is costly. The goal: predict failure type before it happens, monitor for data drift as operating conditions change, and explain why the model flags a given failure risk.

Pipeline Stages

1. Data Validation & EDA

Defined a Pandera schema enforcing domain constraints (e.g., torque 3–80 Nm, rotational speed 1000–2900 rpm) across three datasets: train, current, and stress
Found severe class imbalance in the training set: "No Failure" accounts for 6,762 of 6,993 records; the rarest failure class has only 30 samples
Engineered two physics-based features: mechanical power (Torque × ω) and temperature differential (Process temp − Air temp)

2. Experiment Tracking & Model Selection

Applied SMOTE to the training split only (not validation), to avoid leaking synthetic samples into evaluation
Trained and logged 4 models via MLflow (Logistic Regression, Random Forest, XGBoost, LightGBM), tracking macro F1, weighted F1, accuracy, and per-class F1
Random Forest won on macro F1 (0.736)
Ran a 30-trial Optuna search (TPE sampler) on XGBoost, improving macro F1 from 0.727 → 0.740, then registered the tuned model to the MLflow Model Registry under a production alias

3. Drift Detection & Monitoring

Used Evidently to compare a normal-operation batch and a simulated high-stress batch against the training distribution
No drift detected in the normal batch; 3 of 5 features (rotational speed, torque, tool wear) drifted significantly under stress conditions — a leading indicator that the model would degrade under real-world load changes

4. Explainability

Used SHAP (TreeExplainer) to identify the top driver of each failure class individually, rather than a single global ranking:
Tool Wear Failure → Tool wear
Heat Dissipation Failure → Temperature differential
Power Failure → Mechanical power (derived feature)
Overstrain Failure → Tool wear

Key Findings

The derived Power_W feature outranked its raw components (torque, rotational speed) for predicting Power Failure — evidence that domain-informed feature engineering added real signal
Accuracy is a misleading metric here: a model predicting "No Failure" every time would score >96% accuracy while missing every actual failure — macro F1 was used instead
One failure class kept an F1 of ~0.10 even after SMOTE and tuning; the root cause was traced to data scarcity (30 raw samples), not model choice — the fix is more data collection, not more tuning
Drifted features (torque, rotational speed) map directly to increased Power Failure risk under stress, per the SHAP analysis — informing a concrete retraining trigger

Tech Stack

Python · Pandera · MLflow · Optuna · Evidently · SHAP · XGBoost · LightGBM · scikit-learn · imbalanced-learn
