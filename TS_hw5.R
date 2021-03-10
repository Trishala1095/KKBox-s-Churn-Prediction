install.packages("randomForest")
install.packages("xlsx")

library(dplyr)
library(glmnet)
library(ROCR)
library(e1071)
library(caret)
library(aod)
library(ggplot2)
library(randomForest)
library(xlsx)

#Used the previous Lab 8 method to split the data into train and test data
x1 <- read.csv("/Users/trishala/Spring semester 2020/R assignments/kkbox-churn-prediction-challenge/data 2/churn_comp_refresh/train_v2.csv")
x2 <- read.csv("/Users/trishala/Spring semester 2020/R assignments/kkbox-churn-prediction-challenge/data 3/churn_comp_refresh/transactions_v2.csv")

nrow(x1)
nrow(x2)

#Combining all the datasets
dataset1 = merge(x1,x2)
nrow(dataset1)

index <- sample(1:nrow(dataset1),size=0.5*nrow(dataset1))
train = dataset1[index,]
test = dataset1[-index,]
nrow(train)
nrow(test)

#Selecting required features
cols = c("is_churn", "payment_plan_days", "plan_list_price" ,"actual_amount_paid", "is_auto_renew", "transaction_date", "membership_expire_date", "is_cancel")
new_train = select(train, cols)
new_test = select(test, cols)

#new_train$is_churn <- as.factor(new_train$is_churn)
#new_test$is_churn <- as.factor(new_test$is_churn)


#Problem 1 Random Forest

#Random Forest Model
rf1 <- randomForest(is_churn~.,data = new_train,ntree=201,importance=TRUE)
summary(rf1)
plot(rf1)

#AUC Train data

prob_train <- predict(rf1,newdata=new_train[-1],type="response")

pred=rep(0,nrow(new_train))
prob_train[prob_train >= 0.5] = 1
prob_train[prob_train < 0.5] = 0
table(pred,new_train$is_churn)

#area under the curve
AUC_train = performance(prediction(prob_train, new_train[1]), "auc")@y.values[[1]]
print(paste("Area under the curve :",AUC_train))
#0.807092790726202

#AUC Test data

prob_test <- predict(rf1,newdata=new_test[-1],type="response")

pred=rep(0,nrow(new_test))
prob_test[prob_train >= 0.5] = 1
prob_test[prob_train < 0.5] = 0
table(pred,new_test$is_churn)

#area under the curve
AUC_test = performance(prediction(prob_test, new_test[1]), "auc")@y.values[[1]]
print(paste("Area under the curve :",AUC_test))
#0.500811710999248

#Problem 2 Neural Network

str(new_train)
str(new_test)

#Min-Max Normalization of data for converting each value in 0 and 1
# new_train$payment_plan_days <- (new_train$payment_plan_days - min(new_train$payment_plan_days))/(max(new_train$payment_plan_days)-min(new_train$payment_plan_days))
# new_train$plan_list_price <- (new_train$plan_list_price - min(new_train$plan_list_price))/(max(new_train$plan_list_price)-min(new_train$plan_list_price))
# new_train$actual_amount_paid <- (new_train$actual_amount_paid - min(new_train$actual_amount_paid))/(max(new_train$actual_amount_paid)-min(new_train$actual_amount_paid))
# new_train$transaction_date <- (new_train$transaction_date - min(new_train$transaction_date))/(max(new_train$transaction_date)-min(new_train$transaction_date))
# new_train$membership_expire_date <- (new_train$membership_expire_date - min(new_train$membership_expire_date))/(max(new_train$membership_expire_date)-min(new_train$membership_expire_date))
# 
# new_test$payment_plan_days <- (new_test$payment_plan_days - min(new_test$payment_plan_days))/(max(new_test$payment_plan_days)-min(new_test$payment_plan_days))
# new_test$plan_list_price <- (new_test$plan_list_price - min(new_test$plan_list_price))/(max(new_test$plan_list_price)-min(new_test$plan_list_price))
# new_test$actual_amount_paid <- (new_test$actual_amount_paid - min(new_test$actual_amount_paid))/(max(new_test$actual_amount_paid)-min(new_test$actual_amount_paid))
# new_test$transaction_date <- (new_test$transaction_date - min(new_test$transaction_date))/(max(new_test$transaction_date)-min(new_test$transaction_date))
# new_test$membership_expire_date <- (new_test$membership_expire_date - min(new_test$membership_expire_date))/(max(new_test$membership_expire_date)-min(new_test$membership_expire_date))

#Neural Network model 
install.packages("neuralnet")
library(neuralnet)

net1 <- neuralnet(is_churn~.,data=new_train, hidden=1,err.fct = "ce",linear.output = FALSE)
summary(net1)
plot(net1)

detach(package:neuralnet,unload=T) #only for the prediction function
#AUC Train data

prob_train <- predict(net1,newdata=new_test[-1],type="response")

pred=rep(0,nrow(new_train))
prob_train[prob_train >= 0.5] = 1
prob_train[prob_train < 0.5] = 0
table(pred,new_train$is_churn)

#area under the curve
AUC_train = performance(prediction(prob_train, new_test[1]), "auc")@y.values[[1]]
print(paste("Area under the curve :",AUC_train))
#0.5

#AUC Test data

prob_test <- predict(net1,newdata=new_test[-1],type="response")

pred=rep(0,nrow(new_test))
prob_test[prob_test >= 0.5] = 1
prob_test[prob_test < 0.5] = 0
table(pred,new_test$is_churn)

#area under the curve
AUC_test = performance(prediction(prob_test, new_test[1]), "auc")@y.values[[1]]
print(paste("Area under the curve :",AUC_test))
#0.5

