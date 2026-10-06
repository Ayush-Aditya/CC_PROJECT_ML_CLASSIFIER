# Diabetes Risk Assessment: Classifier Comparison

An end-to-end machine learning project that compares five binary classifiers for diabetes mellitus risk assessment and presents the results in a deployable analytics dashboard. The repository contains the training pipeline, the source dataset, pre-computed evaluation results, and a static Next.js frontend for Vercel.

![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-000?style=flat-square&logo=vercel)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Project Objective

The objective is to evaluate how different classification strategies perform on the same diabetes prediction task. The project is intended for model comparison and academic demonstration, not clinical diagnosis. The dashboard makes the trade-offs between discrimination, recall, specificity, stability, interpretability, and training cost visible in one place.

The current benchmark evaluates:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine
4. K-Nearest Neighbors
5. Gradient Boosting using XGBoost

## Dataset

The project uses the Pima Indians Diabetes Dataset, originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases. It contains 768 records from female patients who are at least 21 years old and have Pima Indian heritage. The dataset is a widely used benchmark for binary diabetes classification, but its population is limited and should not be treated as representative of all age groups, ethnicities, or clinical settings.

The target is `Outcome`:

| Class | Meaning | Records | Share |
|---|---|---:|---:|
| 0 | No diabetes | 500 | 65.1% |
| 1 | Diabetes | 268 | 34.9% |

This is a moderately imbalanced dataset. The pipeline preserves the observed class distribution and uses stratification for both the hold-out split and cross-validation. No synthetic oversampling or explicit class weights are applied. Accuracy is therefore interpreted alongside recall, specificity, F1 score, and ROC-AUC.

### Input Features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration from the 2-hour oral glucose tolerance test |
| `BloodPressure` | Diastolic blood pressure in mm Hg |
| `SkinThickness` | Triceps skin fold thickness in mm |
| `Insulin` | 2-hour serum insulin measurement |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | A family-history-based diabetes pedigree score |
| `Age` | Patient age in years |

## Methodology

### 1. Data preparation

The raw dataset represents several biologically impossible measurements as zero. The pipeline treats zero as missing for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`. Each such value is replaced with the median of its feature. `Pregnancies`, `DiabetesPedigreeFunction`, `Age`, and the target are not modified under this rule.

All feature columns are then standardized with `StandardScaler`. Scaling is important for Logistic Regression, SVM, and KNN because these models are sensitive to feature magnitude. It also provides a consistent input representation for comparison across models.

### 2. Train/test protocol

The data is divided into an 80% training set and a 20% test set using a stratified split with `random_state=42`. The resulting benchmark uses 614 training records and 154 held-out test records. The test set is used only for the final reported metrics and ROC curves.

### 3. Classifier rationale

| Model | Approach | Why it is included | Strongest use case |
|---|---|---|---|
| Logistic Regression | Linear probabilistic classifier | Provides a fast, interpretable baseline | A transparent reference model and probability estimate |
| Random Forest | Bagged decision-tree ensemble | Captures non-linear interactions and is relatively robust to noisy inputs | Strong general-purpose tabular performance |
| SVM | Maximum-margin classifier with an RBF kernel | Separates complex boundaries after feature scaling | High-dimensional or non-linear separation |
| KNN | Instance-based majority vote | Makes predictions from nearby training examples | Local pattern recognition and a simple non-parametric comparison |
| XGBoost | Sequential gradient-boosted trees | Builds corrective trees and models complex feature interactions | High-capacity tabular modelling |

The pipeline uses fixed random seeds where supported so that the exported results can be reproduced. If XGBoost is unavailable, the training script falls back to scikit-learn's `GradientBoostingClassifier` so the five-model comparison can still be generated.

### 4. Evaluation

Each model is fitted on the training data, then evaluated on the held-out test set using:

- Accuracy: overall proportion of correct predictions.
- Precision: proportion of predicted positive cases that are positive.
- Recall: proportion of actual positive cases detected by the model.
- F1 score: harmonic mean of precision and recall.
- ROC-AUC: ranking quality across classification thresholds.
- Specificity: proportion of negative cases correctly identified.
- Confusion matrix: true negatives, false positives, false negatives, and true positives.
- Training time: measured in milliseconds for the fitted estimator.

The pipeline also performs five-fold stratified cross-validation on the training set. The dashboard reports each model's mean accuracy and standard deviation across folds to distinguish a strong single split from a stable model.

## Dashboard

The Next.js dashboard is designed as a compact model observability console rather than a marketing page. Its opening view explains the dataset, class balance, preprocessing protocol, train/test split, model strengths, and feature set before presenting the charts.

The dashboard includes:

- Dataset profile with class counts, class percentages, and preprocessing details.
- Model roster with short descriptions, primary strengths, and AUC scores.
- Grouped metric comparison across the five classifiers.
- Overlaid ROC curves with a random-classifier reference line.
- Confusion-matrix panels for every model.
- Tree-based feature-importance charts.
- Five-fold cross-validation means and standard deviations.
- Radar comparison across core performance metrics.
- Detailed results table with best values highlighted.

The static results file at `dashboard/public/data/results.json` is generated by the Python pipeline and imported at build time. The frontend does not require a backend or runtime database.

## Repository Structure

```
.
|-- ml_pipeline/
|   |-- train_models.py           # Training, evaluation, and JSON export
|   |-- diabetes.csv              # Local Pima dataset copy
|   `-- requirements.txt          # Python dependencies
|-- dashboard/
|   |-- app/                      # Next.js App Router and global styles
|   |-- components/               # Dashboard sections and chart components
|   |-- lib/data.ts               # Typed import boundary for results data
|   |-- public/data/logo.png      # Diabetes Risk Lab brand mark
|   |-- public/data/results.json  # Pre-computed dashboard data
|   |-- package.json
|   `-- next.config.mjs           # Static export configuration
|-- vercel.json                   # Vercel build and output configuration
`-- README.md
```

## Local Development

### Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- npm 9 or newer

### Generate model results

From the repository root:

```bash
cd ml_pipeline
python -m pip install -r requirements.txt
python train_models.py
```

The script writes the complete evaluation payload to `dashboard/public/data/results.json`.

### Run the dashboard

```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:3000` in a browser.

### Validate the production build

```bash
cd dashboard
npm run build
```

Because `next.config.mjs` uses `output: 'export'`, the static site is generated in `dashboard/out/`.

## Vercel Deployment

The root `vercel.json` configures Vercel to install dependencies and build the dashboard from the `dashboard/` directory:

- Build command: `cd dashboard && npm install && npm run build`
- Output directory: `dashboard/out`
- Framework: static export

To deploy through the Vercel dashboard, import the GitHub repository and deploy with the checked-in configuration. The project can also be deployed with the Vercel CLI:

```bash
npm install --global vercel
vercel --prod
```

## Limitations and Responsible Use

This project is a model-comparison demonstration using a small, historically collected benchmark dataset. It does not provide medical advice, diagnosis, treatment recommendations, or a validated clinical decision-support system. The reported scores depend on the fixed split, preprocessing choices, and dataset population. Any real-world deployment would require external validation, clinical review, calibration analysis, fairness assessment, privacy controls, monitoring, and regulatory review.

## License

This project is licensed under the MIT License.
