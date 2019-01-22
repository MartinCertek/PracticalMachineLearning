Load data

set.seed(12345)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

Partioning the training set into two

inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
TrainDatas <- training[inTrain, ]
TestDatas <- training[-inTrain, ]
dim(TrainDatas); dim(TestDatas)
## [1] 11776   160
## [1] 7846  160
Cleaning the data
Remove NearZeroVariance variables

rzv <- nearZeroVar(TrainDatas, saveMetrics=TRUE)
TrainDatas <- TrainDatas[,rzv$rzv==FALSE]

rzv<- nearZeroVar(TestDatas,saveMetrics=TRUE)
TestDatas <- TestDatas[,rzv$rzv==FALSE]
Remove the first column of the TrainDatas data set

TrainDatas <- TrainDatas[c(-1)]
Clean variables with more than 60% NA

trainingV3 <- TrainDatas
for(i in 1:length(TrainDatas)) {
    if( sum( is.na( TrainDatas[, i] ) ) /nrow(TrainDatas) >= .7) {
        for(j in 1:length(trainingV3)) {
            if( length( grep(names(TrainDatas[i]), names(trainingV3)[j]) ) == 1)  {
                trainingV3 <- trainingV3[ , -j]
            }   
        } 
    }
}

# Set back to the original variable name
TrainDatas <- trainingV3
rm(trainingV3)
Transform the TestDatas and testing data sets

clean1 <- colnames(TrainDatas)
clean2 <- colnames(TrainDatas[, -58])  
TestDatas <- TestDatas[clean1]         
testing <- testing[clean2]             

dim(TestDatas)
## [1] 7846   58
dim(testing)
## [1] 20 57
Coerce the data into the same type

for (i in 1:length(testing) ) {
    for(j in 1:length(TrainDatas)) {
        if( length( grep(names(TrainDatas[i]), names(testing)[j]) ) == 1)  {
            class(testing[j]) <- class(TrainDatas[i])
        }      
    }      
}

# To get the same class between testing and TrainDatas
testing <- rbind(TrainDatas[2, -58] , testing)
testing <- testing[-1,]
Prediction with Decision Trees
set.seed(12345)
modFitA1 <- rpart(classe ~ ., data=TrainDatas, method="class")
fancyRpartPlot(modFitA1)

predA1 <- predict(modFitA1, TestDatas, type = "class")
cmtree <- confusionMatrix(predA1, TestDatas$classe)
cmtree

## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2150   60    7    1    0
##          B   61 1260   69   64    0
##          C   21  188 1269  143    4
##          D    0   10   14  857   78
##          E    0    0    9  221 1360
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8789          
##                  95% CI : (0.8715, 0.8861)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8468          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9633   0.8300   0.9276   0.6664   0.9431
## Specificity            0.9879   0.9693   0.9450   0.9845   0.9641
## Pos Pred Value         0.9693   0.8666   0.7809   0.8936   0.8553
## Neg Pred Value         0.9854   0.9596   0.9841   0.9377   0.9869
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2740   0.1606   0.1617   0.1092   0.1733
## Detection Prevalence   0.2827   0.1853   0.2071   0.1222   0.2027
## Balanced Accuracy      0.9756   0.8997   0.9363   0.8254   0.9536


plot(cmtree$table, col = cmtree$byClass, main = paste("Decision Tree Confusion Matrix: Accuracy =", round(cmtree$overall['Accuracy'], 4)))

Prediction with Random Forests
set.seed(12345)
modFitB1 <- randomForest(classe ~ ., data=TrainDatas)
predictionB1 <- predict(modFitB1, TestDatas, type = "class")
cmrf <- confusionMatrix(predictionB1, TestDatas$classe)
cmrf
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    2    0    0    0
##          B    1 1516    0    0    0
##          C    0    0 1367    3    0
##          D    0    0    1 1282    1
##          E    0    0    0    1 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9989          
##                  95% CI : (0.9978, 0.9995)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9987   0.9993   0.9969   0.9993
## Specificity            0.9996   0.9998   0.9995   0.9997   0.9998
## Pos Pred Value         0.9991   0.9993   0.9978   0.9984   0.9993
## Neg Pred Value         0.9998   0.9997   0.9998   0.9994   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1932   0.1742   0.1634   0.1837
## Detection Prevalence   0.2846   0.1933   0.1746   0.1637   0.1838
## Balanced Accuracy      0.9996   0.9993   0.9994   0.9983   0.9996
plot(modFitB1)


plot(cmrf$table, col = cmtree$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cmrf$overall['Accuracy'], 4)))


Prediction with Generalized Boosted Regression

set.seed(12345)
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1)

gbmFit1 <- train(classe ~ ., data=TrainDatas, method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE)


gbmFinMod1 <- gbmFit1$finalModel

gbmPredTest <- predict(gbmFit1, newdata=TestDatas)
gbmAccuracyTest <- confusionMatrix(gbmPredTest, TestDatas$classe)
gbmAccuracyTest

## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    4    0    0    0
##          B    1 1512    1    0    0
##          C    0    2 1358    6    0
##          D    0    0    9 1271    0
##          E    0    0    0    9 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9959          
##                  95% CI : (0.9942, 0.9972)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9948          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9960   0.9927   0.9883   1.0000
## Specificity            0.9993   0.9997   0.9988   0.9986   0.9986
## Pos Pred Value         0.9982   0.9987   0.9941   0.9930   0.9938
## Neg Pred Value         0.9998   0.9991   0.9985   0.9977   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1927   0.1731   0.1620   0.1838
## Detection Prevalence   0.2849   0.1930   0.1741   0.1631   0.1849
## Balanced Accuracy      0.9994   0.9979   0.9957   0.9935   0.9993

plot(gbmFit1, ylim=c(0.9, 1))

predictionB2 <- predict(modFitB1, testing, type = "class")
predictionB2

##  2 31  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
