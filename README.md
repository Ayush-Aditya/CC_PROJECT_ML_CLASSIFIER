# 🏥 Diabetes Risk Assessment: ML Classifier Comparison

A comprehensive comparison of **five machine learning classifiers** for diabetes mellitus risk assessment in young adults, presented through an interactive, premium analytics dashboard.

![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-000?style=flat-square&logo=vercel)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 📊 Overview

This project trains and evaluates five popular ML classifiers on the **Pima Indians Diabetes Dataset** and visualizes their performance through a modern, dark-themed analytics dashboard deployable on Vercel.

### Classifiers Compared

| Classifier | Type | Key Strength |
|-----------|------|-------------|
| **Logistic Regression** | Linear | Interpretable baseline |
| **Random Forest** | Ensemble (Bagging) | Robust, handles non-linearity |
| **Support Vector Machine** | Margin-based | Strong generalization |
| **K-Nearest Neighbors** | Instance-based | Captures local patterns |
| **XGBoost** | Ensemble (Boosting) | State-of-the-art accuracy |

### Dashboard Visualizations

- 📈 **Model Performance Comparison** — Grouped bar charts across all metrics
- 📉 **ROC Curves** — Overlaid receiver operating characteristic curves
- 🔲 **Confusion Matrices** — Heatmap-style matrices for each classifier
- 🌳 **Feature Importance** — Horizontal bar charts from tree-based models
- 🔄 **Cross-Validation** — CV score distributions with mean ± std
- 🕸️ **Radar Chart** — Multi-dimensional model comparison
- 📋 **Detailed Results Table** — Full metrics with best-value highlighting

## 🗂️ Project Structure

```
├── ml_pipeline/                  # Python ML pipeline
│   ├── train_models.py           # Model training & evaluation script
│   ├── diabetes.csv              # Pima Indians Diabetes Dataset
│   └── requirements.txt          # Python dependencies
├── dashboard/                    # Next.js frontend dashboard
│   ├── app/                      # App Router pages & layouts
│   ├── components/               # React chart components
│   ├── public/data/results.json  # Pre-computed ML results
│   ├── package.json
│   └── next.config.mjs
├── vercel.json                   # Vercel deployment config
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

### 1. Run the ML Pipeline

```bash
cd ml_pipeline
pip install -r requirements.txt
python train_models.py
```

This trains all 5 classifiers and exports results to `dashboard/public/data/results.json`.

### 2. Run the Dashboard Locally

```bash
cd dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

### 3. Build for Production

```bash
cd dashboard
npm run build
```

Static output is generated in `dashboard/out/`.

## ☁️ Deploy to Vercel

### Option A: One-click deploy
1. Push this repo to GitHub
2. Import the repo on [vercel.com](https://vercel.com)
3. Vercel will auto-detect the config from `vercel.json`
4. Deploy!

### Option B: Vercel CLI
```bash
npm i -g vercel
vercel --prod
```

## 📝 Dataset

**Pima Indians Diabetes Dataset**
- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases
- **Samples**: 768 female patients (≥21 years, Pima Indian heritage)
- **Features**: 8 diagnostic measurements
- **Target**: Binary (diabetes / no diabetes)

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2h OGTT) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (μU/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age (years) |

## 🧪 Methodology

1. **Preprocessing**: Median imputation for biologically impossible zero values, StandardScaler normalization
2. **Split**: 80/20 stratified train/test split (random_state=42)
3. **Evaluation**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Specificity
4. **Validation**: 5-fold stratified cross-validation

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Pipeline | Python, scikit-learn, XGBoost, pandas, NumPy |
| Frontend | Next.js 14, TypeScript, React 18 |
| Charts | Recharts |
| Animations | Framer Motion |
| Styling | Custom CSS (glassmorphism, dark theme) |
| Deployment | Vercel (static export) |

## 📄 License

This project is licensed under the MIT License.
