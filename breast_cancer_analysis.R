#Predict whether the breast tissue sample is cancerous or not, 
#given the measurements of nuclei using different machine learning models



#import libraries

library(caret)
library(ggplot2)

#Import dataset
dataset <- read.csv("data.csv")

#Reading s structure of data
str(dataset)


#Remove id and X columns
dataset <- dataset[2:32]

#Summary of dataset
summary(dataset)

#Encoding the target feature as factor

dataset$diagnosis <-factor(dataset$diagnosis,
                           levels = c('B','M'),
                           labels =c(0,1))

#Analysis and Visualization of the dataset
#Distribution of diagnosis column 

prop.table(table(dataset$diagnosis))
counts<- table(dataset$diagnosis)
barplot(counts)

#Distribution of the features grouped by diagnosis
boxplot(dataset[,2:ncol(dataset)], main='Multiple Box plots')


#Correlation of the features

library(corrplot)
corr_diag<- cor(dataset[,2:ncol(dataset)])
corrplot(corr_diag)


#Splitting dataset into training set and test set
#install.packages(caTools)
library(caTools)
set.seed(123)
split <- sample.split(dataset$diagnosis, SplitRatio = 0.75)
training_set <- subset(dataset,split==TRUE)
test_set <- subset(dataset,split==FALSE)

#Feature Scaling
training_set[2:31] <- scale(training_set[2:31])
test_set[2:31] <- scale(test_set[2:31])

#Model Selection
# Fitting logistic regression to the Training set
#install.packages('e1071')
library(e1071)

classifier <- glm(formula = diagnosis ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results
prob_pred <- predict(classifier, type = 'response', newdata = test_set[-1])
y_pred <- ifelse(prob_pred > 0.5, 1, 0)
cm <- table(test_set[, 1], y_pred)


#Fitting SVM to training set
classifier <- svm(formula = diagnosis ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred <- predict(classifier, newdata = test_set[-1])

# Making the Confusion Matrix
cm <- table(test_set[, 1], y_pred)



# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
y_pred <- knn(train = training_set[, -1],
             test = test_set[, -1],
             cl = training_set[, 1],
             k = 5,
             prob = TRUE)

# Making the Confusion Matrix
cm <- table(test_set[, 1], y_pred)

# Fitting Random Forest Classification to the Training set
#install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier <- randomForest(x = training_set[-1],
                          y = training_set$diagnosis,
                          ntree = 500)

# Predicting the Test set results
y_pred <- predict(classifier, newdata = test_set[-1])

# Making the Confusion Matrix
cm <- table(test_set[, 1], y_pred)

#Fitting Naive Bayes to the Training set
library(e1071)
classifier <- naiveBayes(x = training_set[-1],
                        y = training_set$diagnosis)

# Predicting the Test set results
y_pred <- predict(classifier, newdata = test_set[-1])

# Making the Confusion Matrix
cm <- table(test_set[, 1], y_pred)

#Fitting Decision Tree to the Training set
library(rpart)
classifier <- rpart(formula = diagnosis~.,
                   data = training_set)

# Predicting the Test set results
y_pred<-predict(classifier, newdata = test_set[-1], type = 'class')

# Making the Confusion Matrix
cm <- table(test_set[, 1], y_pred)
