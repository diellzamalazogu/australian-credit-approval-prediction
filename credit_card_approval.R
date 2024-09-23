# Load required libraries
install.packages(c("glmnet", "pamr", "caret", "nnet", "corrplot", "fastDummies", "pROC", "e1071", "klaR"))
library(caret)
library(nnet)
library(corrplot)
library(fastDummies)
library(pROC)
library(e1071)  # For SVM
library(klaR)   # For Naive Bayes

# Load dataset
data <- read.table('australian.dat')

# Rename columns to meaningful names
colnames(data) <- c("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15")

# Preprocess data
# Create dummy variables for categorical columns
data <- fastDummies::dummy_cols(data, select_columns = c('A4', 'A5', 'A6', 'A12'), remove_first_dummy = TRUE)

# Set predictors and response variable
predictor <- subset(data, select = -A15)
response <- data$A15

# Remove near-zero variance predictors
predictor <- predictor[, -nearZeroVar(predictor)]

# Correlation plot for continuous variables
cont_predictor <- predictor[c('A2', 'A3', 'A7', 'A10', 'A13', 'A14')]
correlations <- cor(cont_predictor)
corrplot(correlations, method = "circle", tl.pos = 'n', type = 'upper')

# Apply Box-Cox transformation
trans <- preProcess(predictor, method = c("BoxCox", "center", "scale"))
predictor <- predict(trans, predictor)

# Apply PCA for dimensionality reduction
trans <- preProcess(predictor, method = "pca")
predictor <- predict(trans, predictor)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(response, p = 0.7, list = FALSE)
trainPredictor <- predictor[trainIndex, ]
trainResponse <- response[trainIndex]
testPredictor <- predictor[-trainIndex, ]
testResponse <- response[-trainIndex]

#-----------------------------------------------------------------------------------------------------
# Model 1: Neural Network
#-----------------------------------------------------------------------------------------------------
nnetGrid <- expand.grid(.size = 1, .decay = 0.1)
nnModel <- train(trainPredictor, trainResponse, method = "nnet", tuneGrid = nnetGrid, trControl = trainControl(method = "cv"))

# Predictions for Neural Network
nnPred_train <- predict(nnModel, newdata = trainPredictor)
nnPred_test <- predict(nnModel, newdata = testPredictor)

#-----------------------------------------------------------------------------------------------------
# Model 2: Support Vector Machine (SVM)
#-----------------------------------------------------------------------------------------------------
svmModel <- train(trainPredictor, trainResponse, method = "svmRadial", tuneLength = 10, trControl = trainControl(method = "cv"))
svmPred_train <- predict(svmModel, newdata = trainPredictor)
svmPred_test <- predict(svmModel, newdata = testPredictor)

#-----------------------------------------------------------------------------------------------------
# Model 3: k-Nearest Neighbors (k-NN)
#-----------------------------------------------------------------------------------------------------
knnModel <- train(trainPredictor, trainResponse, method = "knn", tuneLength = 10, trControl = trainControl(method = "cv"))
knnPred_train <- predict(knnModel, newdata = trainPredictor)
knnPred_test <- predict(knnModel, newdata = testPredictor)

#-----------------------------------------------------------------------------------------------------
# Model 4: Naive Bayes
#-----------------------------------------------------------------------------------------------------
naiveModel <- train(trainPredictor, trainResponse, method = "nb", tuneLength = 10, trControl = trainControl(method = "cv"))
nbPred_train <- predict(naiveModel, newdata = trainPredictor)
nbPred_test <- predict(naiveModel, newdata = testPredictor)

#-----------------------------------------------------------------------------------------------------
# Evaluate and Compare Models
#-----------------------------------------------------------------------------------------------------

# Function to print performance metrics
evaluate_model <- function(model_name, trainPred, testPred, trainResponse, testResponse) {
  trainResults <- postResample(trainPred, trainResponse)
  testResults <- postResample(testPred, testResponse)
  cat(paste("Performance of", model_name, "Model:\n"))
  cat("Train Accuracy:", trainResults[1], "\n")
  cat("Test Accuracy:", testResults[1], "\n\n")
}

# Neural Network performance
evaluate_model("Neural Network", nnPred_train, nnPred_test, trainResponse, testResponse)

# SVM performance
evaluate_model("SVM", svmPred_train, svmPred_test, trainResponse, testResponse)

# k-NN performance
evaluate_model("k-NN", knnPred_train, knnPred_test, trainResponse, testResponse)

# Naive Bayes performance
evaluate_model("Naive Bayes", nbPred_train, nbPred_test, trainResponse, testResponse)

#-----------------------------------------------------------------------------------------------------
# Plot ROC Curve for each model (Train)
#-----------------------------------------------------------------------------------------------------

# ROC for Neural Network
roc_nn_train <- roc(response = trainResponse, predictor = as.numeric(nnPred_train))
roc_nn_test <- roc(response = testResponse, predictor = as.numeric(nnPred_test))

# ROC for SVM
roc_svm_train <- roc(response = trainResponse, predictor = as.numeric(svmPred_train))
roc_svm_test <- roc(response = testResponse, predictor = as.numeric(svmPred_test))

# ROC for k-NN
roc_knn_train <- roc(response = trainResponse, predictor = as.numeric(knnPred_train))
roc_knn_test <- roc(response = testResponse, predictor = as.numeric(knnPred_test))

# ROC for Naive Bayes
roc_nb_train <- roc(response = trainResponse, predictor = as.numeric(nbPred_train))
roc_nb_test <- roc(response = testResponse, predictor = as.numeric(nbPred_test))

# Plot ROC Curves
plot(roc_nn_train, col = "red", main = "Train ROC Curves for All Models")
lines(roc_svm_train, col = "blue")
lines(roc_knn_train, col = "green")
lines(roc_nb_train, col = "purple")
legend("bottomright", legend = c("Neural Network", "SVM", "k-NN", "Naive Bayes"), col = c("red", "blue", "green", "purple"), lty = 1)
