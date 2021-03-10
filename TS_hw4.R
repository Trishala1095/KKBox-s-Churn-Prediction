library(caret)
library(dplyr)
library(aod)
library(ROCR)

x1 <- read.csv("/Users/trishala/Spring semester 2020/R assignments/kkbox-churn-prediction-challenge/data 2/churn_comp_refresh/train_v2.csv")
x2 <- read.csv("/Users/trishala/Spring semester 2020/R assignments/kkbox-churn-prediction-challenge/data 3/churn_comp_refresh/transactions_v2.csv")

nrow(x1)
nrow(x2)

# x2$is_churn <- x1$is_churn
# x2$is_churn
# head(x2)
# str(x2)
# nrow(x2)
# x <- training_data[sample(1:nrow(training_data), 500,replace=FALSE),]

#Combining all the datasets
dataset1 = merge(x1,x2)
nrow(dataset1)

# #Transforming the dataset's categorical values
# dataset1$msno <- factor(dataset1$msno)
# dataset1$payment_method_id <- factor(dataset1$payment_method_id)
# dataset1$payment_plan_days <- factor(dataset1$payment_plan_days)
# dataset1$plan_list_price <- factor(dataset1$plan_list_price)
# dataset1$actual_amount_paid <- factor(dataset1$actual_amount_paid)
# dataset1$is_auto_renew<- factor(dataset1$is_auto_renew)
# dataset1$transaction_date <- factor(dataset1$transaction_date)
# dataset1$membership_expire_date <- factor(dataset1$membership_expire_date)
# dataset1$is_cancel <- factor(dataset1$is_cancel)
# dataset1$is_churn <- factor(dataset1$is_churn)

#check whether there is a NA value, if present remove them
nrow(dataset1[is.na(dataset1$msno)|is.na(dataset1$payment_method_id)|is.na(dataset1$payment_plan_days)|is.na(dataset1$actual_amount_paid)|is.na(dataset1$is_auto_renew)|is.na(dataset1$transaction_date)|is.na(dataset1$membership_expire_date)|is.na(dataset1$is_cancel)|is.na(dataset1$is_churn),])
#nrow(training_data[!(is.na(training_data$msno)|is.na(training_data$is_churn)),])

#Problem 1
#Sampling the data; Divide the data in test data and train data as per the question
index <- sample(1:nrow(dataset1),size=0.5*nrow(dataset1))
train = dataset1[index,]
test = dataset1[-index,]
nrow(train)
nrow(test)

#Selecting required features
cols = c("is_churn", "payment_plan_days", "plan_list_price" ,"actual_amount_paid", "is_auto_renew", "transaction_date", "membership_expire_date", "is_cancel")
new_train = select(train, cols)
new_test = select(test, cols)

#Logistic regression model building
logistic <- glm(is_churn~., data = new_train, family = binomial)
summary(logistic)

prob <- predict(logistic,newdata=new_test[-1],type="response") 
#head(new_test)

#Confusion matrix and accuracy
pred=rep(0,nrow(new_test))
prob[prob >= 0.5] = 1
prob[prob < 0.5] = 0
table(pred,new_test$is_churn)

#Performance testing on the plot
accuracy = mean(new_test$is_churn == prob) 
print(paste("Accuracy of model :",accuracy))

#Classification error
c_error = 1 - accuracy
print(paste("Classification error :",c_error))

#Finding the Area under the curve 
AUC = performance(prediction(prob, new_test[1]), "auc")@y.values[[1]]
print(paste("Area under the curve :",AUC))

#Plotting the AUC
plot(performance(prediction(prob, new_test[1]), "tpr","fpr"))

# Problem 2
#Cross validation method
con <- trainControl(method = "cv", number = 10)
new_train$is_churn <- as.factor(new_train$is_churn)
logistic2 <- train(is_churn~., data=new_train, trControl=con, method="glm")
summary(logistic2)

#Finding the accuracy
prob2 = predict(logistic2, newdata=new_test[-1])
accuracy1 = mean(new_test$is_churn == prob2) 
print(paste("Accuracy of model :",accuracy1))

#Classification error
c_error1 = 1 - accuracy
print(paste("Classification error :",c_error1))

print("The classification error is coming equal to the classification error in the question 1 i.e; 0.0772007250652806")


# Problem 3

#with the kaggle dataset

x3 <- read.csv("/Users/trishala/Spring semester 2020/R assignments/kkbox-churn-prediction-challenge/data/churn_comp_refresh/sample_submission_v2.csv")
dataset2 = merge(x3,x2)
cols = c("is_churn", "payment_plan_days", "plan_list_price" ,"actual_amount_paid", "is_auto_renew", "transaction_date", "membership_expire_date", "is_cancel")
new_train2 = select(dataset1, cols)
new_test2 = select(dataset1, cols)

logistic3 = glm(new_train2$is_churn ~ ., data = new_train2, family = "binomial")
summary(logistic3)
prob3 = predict(logistic3, newdata=new_test2[-1], type="response")
prob3[prob3 >= 0.5] = 1
prob3[prob3 < 0.5] = 0

accuracy2 = mean(new_test$is_churn == prob3) 
print(paste("Accuracy of model :",accuracy2))

c_error2 = 1 - accuracy
print(paste("Classification error :",c_error2))
