# Hazardous Asteroid Classification

This project uses machine learning to classify potentially hazardous asteroids using two asteroid-related datasets. The notebook compares multiple supervised learning models on both an asteroid orbital dataset and a close-approach dataset.

The goal is to predict whether an asteroid is hazardous based on orbital, physical, and close-approach features.

## Project File

- `164Project.ipynb` — Main Google Colab / Jupyter Notebook containing dataset loading, preprocessing, model training, evaluation, and hyperparameter tuning.

## Datasets Used

This project uses two Kaggle datasets:

1. **Asteroid Dataset**
   - Loaded from: `sakhawat18/asteroid-dataset`
   - Main target column: `pha`
   - The target represents whether an asteroid is potentially hazardous.

2. **NeoWS Hazardous Asteroid Dataset**
   - Loaded from: `alvinb/neows-hazardous-asteroid-dataset`
   - Main target column: `Hazardous`
   - The dataset is made up of multiple close-approach CSV files covering different date ranges.

## Tools and Libraries

The notebook uses:

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- kagglehub
- Google Colab

Main scikit-learn tools used:

- `train_test_split`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `LogisticRegression`
- `KNeighborsClassifier`
- `MLPClassifier`
- `GridSearchCV`

## Workflow

The notebook follows this general process:

1. Load both asteroid datasets.
2. Combine the close-approach CSV files into one dataframe.
3. Explore missing values.
4. Drop columns with excessive missing data.
5. Remove duplicate rows and empty rows/columns.
6. Remove low-value identifier columns.
7. Convert categorical features using mapping and one-hot encoding.
8. Convert date columns into year, month, day, and day-of-year features.
9. Remove redundant measurement columns.
10. Create stratified train, validation, and test splits.
11. Apply median imputation.
12. Apply standard scaling.
13. Train baseline models.
14. Evaluate models using validation metrics.
15. Tune hyperparameters with `GridSearchCV`.
16. Compare tuned model performance across both datasets.

## Models Compared

The project compares three supervised learning models:

| Model | Description |
|---|---|
| Logistic Regression | Linear classification model with balanced class weights |
| K-Nearest Neighbors | Distance-based classifier |
| MLP Classifier | Basic neural network classifier |

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Because the datasets are imbalanced, F1 score and recall are especially important when comparing model performance.

## Baseline Results Summary

### Asteroid Dataset

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9903 | 0.1944 | 1.0000 | 0.3256 | 0.9963 |
| KNN | 0.9970 | 0.0000 | 0.0000 | 0.0000 | 0.7833 |
| MLP | 0.9970 | 0.0000 | 0.0000 | 0.0000 | 0.9950 |

### Close-Approach Dataset

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9377 | 0.6876 | 0.9538 | 0.7991 | 0.9895 |
| KNN | 0.9073 | 0.7014 | 0.5000 | 0.5838 | 0.9175 |
| MLP | 0.9920 | 0.9668 | 0.9718 | 0.9693 | 0.9995 |

## Tuned Results Summary

### Asteroid Dataset

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9923 | 0.2333 | 1.0000 | 0.3784 | 0.9979 |
| KNN | 0.9967 | 0.0000 | 0.0000 | 0.0000 | 0.7836 |
| MLP | 0.9973 | 0.0000 | 0.0000 | 0.0000 | 0.3503 |

### Close-Approach Dataset

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9380 | 0.6889 | 0.9538 | 0.8000 | 0.9895 |
| KNN | 0.9087 | 0.6801 | 0.5615 | 0.6152 | 0.8891 |
| MLP | 0.9913 | 0.9550 | 0.9795 | 0.9671 | 0.9992 |

## Key Findings

- The close-approach dataset produced stronger classification results than the asteroid dataset.
- The MLP model performed best on the close-approach dataset, reaching the highest F1 score and ROC-AUC.
- Logistic Regression performed best on the asteroid dataset because it successfully identified hazardous asteroids, while KNN and MLP struggled with the minority class.
- Accuracy alone was not enough to judge model quality because some models achieved high accuracy while failing to detect hazardous asteroids.

## How to Run

This notebook was designed to run in Google Colab.

1. Open `164Project.ipynb` in Google Colab or Jupyter Notebook.
2. Make sure the required packages are installed.
3. Run the notebook cells in order.
4. The datasets will be downloaded using `kagglehub` if they are not already available.
5. Review the output tables for baseline and tuned model performance.

## Required Packages

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
kagglehub
