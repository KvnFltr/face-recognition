[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/KvnFltr/face-recognition/blob/main/notebooks/facial_recognition_code_only.ipynb)

# Bias-aware Face Recognition (Notebook)

This repo hosts a single Jupyter notebook implementing a face recognition pipeline:
**detection/landmarks (dlib + HOG) → feature extraction (CNN/Keras/TensorFlow) → classification (SVM / kNN / Logistic Regression, scikit-learn)**,
with basic evaluation.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab  # or jupyter notebook
