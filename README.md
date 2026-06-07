# Glassbox ML

**Watch machine learning algorithms actually learn.**

Classic ML algorithms, implemented **from scratch** — no scikit-learn, no PyTorch — each paired with an **interactive, animated** browser exhibit. The math you read is the math running on screen. The black box, made glass.

> The missing third kind of ML resource: not pure theory, not `model.fit()` — you *watch* the algorithm train, step by step.

---

## ✨ What makes it different

| | |
|---|---|
| ⌨️ **From scratch** | Pure NumPy (Python canon) and vanilla JS (the animations). No ML libraries hiding the work. |
| 🎬 **Animated** | Loss curves drop, decision boundaries rotate, clusters form. Scrub through any training epoch. |
| ✅ **Verified vs scikit-learn** | Every from-scratch model is benchmarked against the library, so you know it's the real thing — not a teaching simplification. |

---

## 📚 Curriculum

A path, not a catalogue — each level builds the intuition the next one needs.

| Level | Topic | Algorithms |
|------|-------|-----------|
| **1** | Foundations | ✅ Linear Regression · ✅ Logistic Regression · Gradient Descent |
| **2** | Distance & Geometry | k-Nearest Neighbors · k-Means · PCA |
| **3** | Probability | Naive Bayes |
| **4** | Trees | Decision Tree · Random Forest |
| **5** | Boosting | AdaBoost · Gradient Boosting · XGBoost · LightGBM |

✅ = live exhibit. More landing continuously.

---

## 🚀 Run it

**The site** — no build step, just open the files:

```bash
# from the repo root
python -m http.server 8000
# then visit http://localhost:8000
```

Or simply double-click `index.html`.

**The Python canon + scikit-learn benchmarks:**

```bash
pip install numpy scikit-learn
cd ml_algos
python test_linear_regression.py
python test_logistic_regression.py
```

Each test proves the from-scratch implementation matches scikit-learn on the same data.

---

## 🗂️ Layout

```
glassbox-ml/
├── index.html                 # landing page + curriculum
├── linear-regression.html     # animated exhibit
├── logistic-regression.html   # animated exhibit
└── ml_algos/                  # the "from scratch" Python canon
    ├── linear_regression.py
    ├── logistic_regression.py
    ├── test_linear_regression.py
    └── test_logistic_regression.py
```

Each algorithm has two surfaces of the same idea: the **Python canon** (reference + benchmark) and the **JS exhibit** (the animation).

---

## 🛠️ Built with

Vanilla JavaScript + HTML Canvas (no framework, no bundler) · NumPy · scikit-learn (for validation only).

---

*An open educational project. Built from scratch, on purpose.*
