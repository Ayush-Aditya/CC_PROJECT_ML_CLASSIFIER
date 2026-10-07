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

The project uses the CDC BRFSS 2015 Diabetes Health Indicators Dataset, a large population-health benchmark derived from the Behavioral Risk Factor Surveillance System. The source contains 253,680 records and 21 numeric health, lifestyle, demographic, and access-to-care indicators. To keep the original model comparison practical while avoiding a tiny benchmark, the pipeline selects exactly 10% of the cleaned source rows using `random_state=42`.

The dashboard benchmark therefore contains 25,368 sampled records. Its target is `Diabetes_binary`:

| Class | Meaning | Records | Share |
|---|---|---:|---:|
| 0 | No diabetes | 21,874 | 86.2% |
| 1 | Diabetes | 3,494 | 13.8% |

This dataset is substantially imbalanced toward the negative class. The random sample preserves the source distribution approximately, and the pipeline uses stratification for the 80/20 hold-out split and five-fold cross-validation. The models use cost-sensitive training: balanced class weights for Logistic Regression, Random Forest, and SVM; positive-class weighting for XGBoost; and distance weighting for KNN. No synthetic oversampling is applied. Accuracy is therefore interpreted alongside balanced accuracy, recall, specificity, F1 score, and ROC-AUC.

### Input Features

| Feature | Description |
|---|---|
| `HighBP` | High blood pressure indicator |
| `HighChol` | High cholesterol indicator |
| `CholCheck` | Cholesterol check within the last five years |
| `BMI` | Body mass index |
| `Smoker` | Smoking history indicator |
| `Stroke` | History of stroke indicator |
| `HeartDiseaseorAttack` | History of coronary heart disease or heart attack |
| `PhysActivity` | Physical activity outside work in the past 30 days |
| `Fruits` | Fruit consumption indicator |
| `Veggies` | Vegetable consumption indicator |
| `HvyAlcoholConsump` | Heavy alcohol consumption indicator |
| `AnyHealthcare` | Health care coverage indicator |
| `NoDocbcCost` | Unable to see a doctor because of cost |
| `GenHlth` | Self-reported general health rating |
| `MentHlth` | Number of days mental health was not good |
| `PhysHlth` | Number of days physical health was not good |
| `DiffWalk` | Difficulty walking or climbing stairs |
| `Sex` | Respondent sex category |
| `Age` | Age category |
| `Education` | Education level category |
| `Income` | Income category |

## Methodology

### 1. Data preparation

The source indicators are already numeric. The pipeline selects the required 21 predictors, removes rows with missing values, samples 10% with a fixed random seed, and standardizes the sampled features with `StandardScaler`. The sample operation occurs before the stratified train/test split, and the scaler is fitted only on the training partition.

All feature columns are then standardized with `StandardScaler`. Scaling is important for Logistic Regression, SVM, and KNN because these models are sensitive to feature magnitude. It also provides a consistent input representation for comparison across models.

### 2. Train/test protocol

The sampled data is divided into an 80% training set and a 20% test set using a stratified split with `random_state=42`. The resulting benchmark uses 20,294 training records and 5,074 held-out test records. The test set is used only for the final reported metrics and ROC curves.

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

- Accuracy: overall proportion of correct predictions; this can be misleading for an imbalanced target.
- Balanced accuracy: average of positive-class recall and negative-class specificity.
- Precision: proportion of predicted positive cases that are positive.
- Recall: proportion of actual positive cases detected by the model.
- F1 score: harmonic mean of precision and recall.
- ROC-AUC: ranking quality across classification thresholds.
- Specificity: proportion of negative cases correctly identified.
- Confusion matrix: true negatives, false positives, false negatives, and true positives.
- Training time: measured in milliseconds for the fitted estimator.

The pipeline also performs five-fold stratified cross-validation on the training set. The dashboard reports each model's mean accuracy and standard deviation across folds to distinguish a strong single split from a stable model. For screening-oriented interpretation, recall, balanced accuracy, F1 score, and the confusion matrix receive more attention than raw accuracy.

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
|   |-- diabetes_brfss2015.csv    # Local CDC BRFSS dataset copy
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
