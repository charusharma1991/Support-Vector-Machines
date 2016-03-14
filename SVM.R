library(e1071)
traindata = read.csv("TrainData.csv")  
test = read.csv("TestInternshipStudent.csv")
col1 <- ncol(traindata)
traindata <- traindata[,c(3:col1)]
col <- ncol(traindata)
trainlabel <- traindata[,col]
traindata <- traindata[,c(1:col-1)]
trainlabel <- as.factor(trainlabel)
test <- test[,c(3:ncol(test))]
svm_model <- svm(x=traindata,y=trainlabel)
p <- predict(svm_model,test)
write.table(p,file="svmPredict.csv")
