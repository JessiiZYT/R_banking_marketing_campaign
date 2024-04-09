#Import packages and dataset, then have overview
install.packages("dplyr")
library(dplyr)
library(readr)
library(magrittr)
library(class)
library(ggplot2)
library(caret)

# Reading Data
data <- read.csv('/Users/quanzhou/Desktop/bank/bank-full.csv', sep = ';')
head(data)
tail(data)
# Viewing data 
summary(data)
str(data)

#checking if there's any null value in the dataset
sum(is.na(data))
dim(data)
#after checking, there is no current null value in the dataset

#select relevant columns for further cleaning
data <- data %>% dplyr::select(-duration,-month)
# we eliminated variables 'duration' because it is only used for benchmark and we don't know duration until the call, 
##  so it will not be useful in the model building
# we also eliminated month because it will cause numerous dummy variables to be generated. 
##  We wish to avoid such situation

# Finding unique catagories and counts of each catagories in each variables
unique_counts <- lapply(data, function(column) {
  counts = table(column)
  return(counts)
})
unique_counts

# Convert categorical variables to factors and check for missing values
data <- data %>% mutate_if(is.character, as.factor) %>% na.omit()  # Remove rows with any NA values in any column

# Checking data again to make sure we only have factor and numeric data
str(data)

###############################################################################################################
#####################################Exploratory Data Analysis ################################################
###############################################################################################################

# Plot Histogram to 6 Numerical Variable
hist(data$age, main = "Histogram of Age", xlab = "Age")
hist(data$balance, main = "Histogram of Balance", xlab = "Balance",xlim = c(0, 35000))

#plot bar plot to 10 Categorical Variable
# Create a bar plot using ggplot2
ggplot(data = data, aes(x = job)) +
  geom_bar() +  
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + # Rotate x-axis labels
  labs(title = "Bar Plot of Jobs", x = "Job Type", y = "Frequency")

# Bar plot for 'marital'
ggplot(data, aes(x = marital)) +
  geom_bar() +
  labs(title = "Bar Plot of Marital Status", x = "Marital Status", y = "Frequency") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Bar plot for 'education'
ggplot(data, aes(x = education)) +
  geom_bar() +
  labs(title = "Bar Plot of Education", x = "Education Level", y = "Frequency") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Bar plot for 'default'
ggplot(data, aes(x = default)) +
  geom_bar() +
  labs(title = "Bar Plot of Credit Default", x = "Credit in Default", y = "Frequency") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Bar plot for 'housing'
ggplot(data, aes(x = housing)) +
  geom_bar() +
  labs(title = "Bar Plot of Housing Loan", x = "Housing Loan", y = "Frequency") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Bar plot for 'loan'
ggplot(data, aes(x = loan)) +
  geom_bar() +
  labs(title = "Bar Plot of Personal Loan", x = "Personal Loan", y = "Frequency") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#Explore relationships among multiple variables.
# Ensure that the 'y' variable is a factor
data$y <- factor(data$y, levels = c("no", "yes"), labels = c("No", "Yes"))

# 1. Age and Subscription
ggplot(data, aes(x = age, y = y)) +
  geom_jitter(width = 0.1, height = 0.1, alpha = 0.5) +
  labs(title = "Age vs Term Deposit Subscription", x = "Age", y = "Subscription") +
  theme_minimal()

# 2. Job Type and Subscription
ggplot(data, aes(x = job, fill = y)) +
  geom_bar(position = "dodge") +
  labs(title = "Job Type vs Term Deposit Subscription", x = "Job Type", y = "Count") +
  theme(axis.text.x = element_text(angle = 65, hjust = 1))

# 3. Marital Status and Financial Decisions
ggplot(data, aes(x = marital, fill = y)) +
  geom_bar(position = "dodge") +
  labs(title = "Marital Status vs Term Deposit Subscription", x = "Marital Status", y = "Count") +
  theme(axis.text.x = element_text(angle = 65, hjust = 1))

# 4. Education and Subscription Rates
ggplot(data, aes(x = education, fill = y)) +
  geom_bar(position = "dodge") +
  labs(title = "Education Level vs Term Deposit Subscription", x = "Education Level", y = "Count") +
  theme(axis.text.x = element_text(angle = 65, hjust = 1))

# 5. Credit Default History and Subscription
ggplot(data, aes(x = default, fill = y)) +
  geom_bar(position = "dodge") +
  labs(title = "Credit Default vs Term Deposit Subscription", x = "Credit Default", y = "Count") +
  theme(axis.text.x = element_text(angle = 65, hjust = 1))

# 6. Loan Status and Subscription
ggplot(data, aes(x = loan, fill = y)) +
  geom_bar(position = "dodge") +
  labs(title = "Personal Loan vs Term Deposit Subscription", x = "Personal Loan", y = "Count") +
  theme(axis.text.x = element_text(angle = 65, hjust = 1))

###############################################################################################################
##################################### Model Construction ######################################################
###############################################################################################################

########################################### Splitting data ####################################################
# spliting the data 
set.seed(123)  # for reproducibility
# Determin sample size, we want to use 2/3 to train our data
trainIndex <- sample(1:nrow(data), 2/3 * nrow(data))
train_data <- data[trainIndex,]
# Checking for the training data 
head(train_data)
dim(train_data)
# Creating Testing Data
test_data <- data[-trainIndex,]
# Checking for the testing data
head(test_data)
dim(test_data)
###############################################################################################################

##################################### Logitstic Regression ####################################################
#fit logistic model
logit.model <- glm(y ~ ., data = train_data, family = binomial())
# model summary
summary(logit.model)
# make prediction
pred.logit <- predict(logit.model, newdata = test_data, type = "response")
pred.logit.class <- ifelse(pred.logit > 0.5, "yes", "no")
#Confusion Matrix
com.mat.log <- table(pred.logit.class, test_data$y)
# Calculate Accuracy
accuracy.log <- sum(diag(com.mat.log))/sum(com.mat.log)
print(print(paste("Logistic Accuracy:", accuracy.log)))

# select best model(stepwise forward)
#install.packages("MASS")
library(MASS)
# Initialize a Null Model and Full Model
nullModel <- glm(y ~ 1, data = train_data, family = binomial)
fullModel <- glm(y ~ ., data = train_data, family = binomial)
#Forward Stepwise Selection
stepModelForward <- stepAIC(nullModel, scope = list(lower = nullModel, upper = fullModel), direction = "forward")
#Backward Stepwise Selection
stepModelBackward <- stepAIC(fullModel, direction = "backward")
#Both Directions Stepwise Selection (Bidirectional)
stepModelBoth <- stepAIC(fullModel, direction = "both")
#Summary model
summary(stepModelBoth)

# Predict on training set
predictionsTrain <- predict(stepModelBoth, newdata = train_data, type = "response")
# Convert probabilities to binary outcomes based on a threshold
predictedClassTrain <- ifelse(predictionsTrain > 0.5, 1, 0)
# Confusion Matrix
confusionMatrixTrain <- table(Predicted = predictedClassTrain, Actual = train_data$y)

# Predict on test set
predictionsTest <- predict(stepModelBoth, newdata = test_data, type = "response")
# Convert probabilities to binary outcomes
predictedClassTest <- ifelse(predictionsTest > 0.5, 1, 0)
# Confusion Matrix
confusionMatrixTest <- table(Predicted = predictedClassTest, Actual = test_data$y)

# Calculate accuracy or other metrics for both sets
accuracyTrain <- sum(diag(confusionMatrixTrain)) / sum(confusionMatrixTrain)
accuracyTest <- sum(diag(confusionMatrixTest)) / sum(confusionMatrixTest)

print(paste("Training Accuracy:", accuracyTrain))
print(paste("Test Accuracy:", accuracyTest))

# Cross_validation
#install.packages("caret")
library(caret)
#Set Up Cross-Validation
control <- trainControl(method = "cv", number = 10)
cvModel <- train(y ~ ., data = train_data, method = "glm", family = "binomial", trControl = control)
#Print cv Model
print(cvModel)
#################################################################################################################


################################################# Decision Tree #################################################
# Tree
# Intall used packages
install.packages("rpart.plot")

library(rpart)
library(rpart.plot)
fit <- rpart(y~., data = train_data, method = 'class', cp=0.001)
rpart.plot(fit, extra = 106)

# make prediction using test data
predictions <- predict(fit, newdata = test_data, type = "class")

# Create a confusion matrix
conf_matrix <- table(predictions, test_data$y)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(accuracy)

#Cross validation
ctrl <- trainControl(method = "cv", number = 5)
cv_fit <- train(y ~ ., data = train_data, method = "rpart", trControl = ctrl)
print(cv_fit)

#trim tree
fit <- rpart(y~., data = train_data, method = 'class', cp=0.001406866)
rpart.plot(fit, extra = 106)
#################################################################################################################


###################################################### KNN ######################################################

### Step 1: creating dummy variables

# Separate predictors and the target variable
## Because we don't want to make our predictor variable a funny variale, so we exclude that column 
train_X <- train_data[, -ncol(train_data)]
train_X_target <- train_data[, ncol(train_data)]
test_X <- test_data[, -ncol(test_data)]
test_X_target <- test_data[, ncol(test_data)]

# One-hot encode features for the training data
train_dummies <- dummyVars(" ~ .", data = train_X)
train_X_transformed <- data.frame(predict(train_dummies, newdata = train_X))

# Apply the same transformation to the test data
test_X_transformed <- data.frame(predict(train_dummies, newdata = test_X))

# Test run: K = 5
set.seed(1)
k <- 5
KNN_Model <- knn(train = train_X_transformed, 
                test = test_X_transformed, 
                cl = train_X_target, 
                k = k)

## Evaluate model performance
confusionMatrix <- table(test_X_target, KNN_Model)
print(confusionMatrix)
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
print(paste("Accuracy:", accuracy))

######################## We are curious to see what the best k is ##############################

############# K evaluation 1: for k with values: (1, 3, 5, 10, 15, 20, 30, 50, 70, 90) #########

# Define a vector of k-values we may be interested 
k_values <- c(1, 3, 5, 10, 15, 20, 30, 50, 70, 90)
# Initialize a dataframe to store k values and their corresponding accuracies
results.1 <- data.frame(k = integer(), accuracy = numeric())

# Loop over different values of k
for (k in k_values) {
  set.seed(1)
  # KNN classification
  KNN_Model <- knn(train = train_X_transformed, 
                   test = test_X_transformed, 
                   cl = train_X_target, 
                   k = k)
  
  # Evaluate model performance
  confusionMatrix <- table(test_X_target, KNN_Model)
  accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)

  # Store the results
  results.1 <- rbind(results.1, data.frame(k = k, accuracy = accuracy))
}

# Print the results
print(results.1)

# Plotting the accuracy vs. k
ggplot(results.1, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point() +
  xlab("Number of Neighbors (k)") +
  ylab("Accuracy") +
  ggtitle("KNN Accuracy vs. Number of Neighbors")
mean(results.1[,2])

################ K evaluation 2: for k with values: 100 to 200, with a step of 10 ##############

# Initialize a dataframe to store k values and their corresponding accuracies
results.2 <- data.frame(k = integer(), accuracy = numeric())

# Loop over different values of k
for (k in seq(100, 200, by = 10)){
  set.seed(1)
  # KNN classification
  KNN_Model <- knn(train = train_X_transformed, 
                   test = test_X_transformed, 
                   cl = train_X_target, 
                   k = k)
  
  # Evaluate model performance
  confusionMatrix <- table(test_X_target, KNN_Model)
  accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
  
  # Store the results
  results.2 <- rbind(results.2, data.frame(k = k, accuracy = accuracy))
}

# Print the results
print(results.2)

# Plotting the accuracy vs. k
ggplot(results.2, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point() +
  xlab("Number of Neighbors (k)") +
  ylab("Accuracy") +
  ggtitle("KNN Accuracy vs. Number of Neighbors")

