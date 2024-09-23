# Australian Credit Card Approval Prediction

This project uses machine learning models to predict Australian credit card approval based on demographic and financial data. We explored various models and chose the **Neural Network** model for its performance, achieving an AUC of 0.8824 and a Kappa of 0.676 on the test data.

## Project Overview

### Goal
The goal of this project is to predict credit card approval (accepted or rejected) using machine learning models trained on the **Australian Credit Card Approval** dataset from the UCI Machine Learning Repository.

### Dataset
- **Source**: UCI Machine Learning Repository
- **Attributes**: 14 features (8 categorical, 6 continuous) with a target variable:
  - `1`: Accepted
  - `2`: Rejected

### Models Explored
1. **Linear Models**: GLM, PLS, LDA, GLMNET
2. **Non-linear Models**: SVM, k-NN, Naive Bayes, Neural Network
3. **Final Model**: Neural Network (chosen based on ROC, AUC, and Kappa values)

## Preprocessing
- **Dummy Variables** for categorical predictors
- **Near Zero Variance** for removing low-variance predictors
- **Box-Cox Transformation** to handle skewness in continuous variables
- **PCA** for dimensionality reduction, preserving 95% of the variance
- **Handling Missing Data** through mode and mean imputation

## Final Results
The final model is a **Neural Network** with the following parameters:
- **Size**: 1
- **Decay**: 0.1

The model achieved satisfactory results:
- **AUC**: 0.8824
- **Kappa**: 0.676

## Project Files
- **[PDF Report](./Australian%20Credit%20Card%20Approval%20Project%20Report.pdf)**: A detailed report explaining the dataset, preprocessing, models, and results.
- **[R Script](./credit_card_approval.R)**: The R code used for preprocessing, model training, and evaluation.

## Running the Code
To run the R code, install the necessary R packages:

```R
install.packages(c("glmnet", "pamr", "caret", "MASS", "nnet", "corrplot"))

