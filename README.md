# Bias-aware Face Recognition (Notebook)

This repo hosts a single Jupyter notebook implementing a face recognition pipeline:
**detection/landmarks (dlib + HOG) → feature extraction (CNN/Keras/TensorFlow) → classification (SVM / kNN / Logistic Regression, scikit-learn)**,
with basic evaluation.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab  # or jupyter notebook
