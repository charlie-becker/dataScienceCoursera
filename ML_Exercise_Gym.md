---
title: "Machine Learning Exercise - Gym Movements"
author: "Charlie Becker"
date: "4/3/2018"
output: html_document
---




### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways (labeled A-E). More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Load libraries and data

The 'foreach' and 'doMC' packages were loaded to utilize 3of 4 computing cores to speed up processing for this dataset.  

The data contained nearly 20,000 observations of 160 total variables from a csv file, though many variables had significant amounts of NA values, missing or blank values, and error values by trying to divide by zero.  To begin cleaning the data - for both the test and train - all missing and error values were converted to NA's and then all NA values summed across columns/variables.  Any column that had a majority of NA's was discarded.  Lastly, all columns that were not relevant predictors (such as names/times) were removed, leaving us with a total of 52 predictor variables.  

Since the dataset is relatively large, and computational time will be on the order of hours - a straight randomized 50/50 split of the "training" set will be used for cross validation.


```r
library(caret)
library(foreach)
library(doMC)
registerDoMC(cores = 3)

trn <- read.csv("/Users/charlesbecker/Downloads/pml-training.csv",na.strings=c("NA","#DIV/0!",""))
tst <- read.csv("/Users/charlesbecker/Downloads/pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

sumNA <- function(x) sum(is.na(x))

NA_sum_trn <- apply(trn, 2, sumNA)
NA_sum_tst <- apply(tst, 2, sumNA)

NA_row_train <- which(NA_sum_trn > nrow(trn)*.5)
NA_row_test <- which(NA_sum_tst > nrow(tst)*.5)

training <- trn[,-NA_row_train]
testing <- tst[,-NA_row_test]

training <- training[,-(1:7)]
testing <- testing[,-(1:7)]

samp <- sample(nrow(training), nrow(training)/2)
new_train <- training[samp,]
cv_data <- training[-samp,]
```

### Build the models

A varity of models will be trained on the same training data and tested on the cross validation set to see error rates.  Classification models included are:  Rpart decision tree, Naive Bayes, Stochastic Gradient Boosting, Neural network, Logistical Regression Boosting, K-Nearest Neighbor, and Random Forest.  All models will be run under the "caret" package wrapper.


```r
mod_rpart <- train(classe ~ ., method = "rpart", data = new_train)
mod_nb <- train(classe ~ ., method = "nb", data = new_train)
mod_gbm <- train(classe ~ ., method = "gbm", data = new_train, verbose = F)
mod_nnet <- train(classe ~ ., method = "nnet", data = new_train)
mod_logit <- train(classe ~ ., method = "LogitBoost", data = new_train)
mod_knn <- train(classe ~ ., method = "knn", data = new_train)
mod_rf <- train(classe ~ ., method = "rf", data = new_train, prox = T)
```

### Predict on the cross validation data and report error rates


```r
table(p_rpart, cv_data$classe)
```

```
##        
## p_rpart    A    B    C    D    E
##       A 2511  795  764  734  267
##       B   57  637   59  270  245
##       C  212  445  910  619  473
##       D    0    0    0    0    0
##       E   11    0    0    0  802
```

```r
confusionMatrix(p_rpart, cv_data$classe)$overall[1]
```

```
##  Accuracy 
## 0.4953623
```

```r
table(p_nb, cv_data$classe)
```

```
##     
## p_nb    A    B    C    D    E
##    A 1957  139  111   90   50
##    B  113 1273  137    7  199
##    C  351  290 1336  314   99
##    D  355  168  146 1149   59
##    E   15    7    3   63 1380
```

```r
confusionMatrix(p_nb, cv_data$classe)$overall[1]
```

```
##  Accuracy 
## 0.7231679
```

```r
table(p_nnet, cv_data$classe)
```

```
##       
## p_nnet    A    B    C    D    E
##      A 1642  189  197   49  123
##      B   50  292  153   72  218
##      C  573  712 1209  627  750
##      D  526  684  173  875  696
##      E    0    0    1    0    0
```

```r
confusionMatrix(p_nnet, cv_data$classe)$overall[1]
```

```
##  Accuracy 
## 0.4095403
```

```r
table(p_gbm, cv_data$classe)
```

```
##      
## p_gbm    A    B    C    D    E
##     A 2736   70    0    3    3
##     B   37 1749   64    5   26
##     C   11   51 1645   61   15
##     D    4    3   23 1543   24
##     E    3    4    1   11 1719
```

```r
confusionMatrix(p_gbm, cv_data$classe)$overall[1]
```

```
##  Accuracy 
## 0.9572928
```

```r
table(p_logit, cv_data$classe)
```

```
##        
## p_logit    A    B    C    D    E
##       A 2371  175   33   28   12
##       B   62 1339  117   18   53
##       C   25   53 1146   51   35
##       D   39   20   65 1275   40
##       E    7   12   14   35 1417
```

```r
confusionMatrix(p_logit, cv_data$classe)$overall[1]
```

```
##  Accuracy 
## 0.8941009
```

```r
table(p_knn, cv_data$classe)
```

```
##      
## p_knn    A    B    C    D    E
##     A 2628  135   32   34   39
##     B   39 1552   62   18  117
##     C   40  100 1535  170   47
##     D   73   48   55 1377   73
##     E   11   42   49   24 1511
```

```r
confusionMatrix(p_knn, cv_data$classe)$overall[1]
```

```
##  Accuracy 
## 0.8768729
```

```r
table(p_rf, cv_data$classe)
```

```
##     
## p_rf    A    B    C    D    E
##    A 2785   19    0    0    0
##    B    4 1854   17    0    2
##    C    1    4 1707   15    2
##    D    0    0    9 1607    3
##    E    1    0    0    1 1780
```

```r
confusionMatrix(p_rf, cv_data$classe)$overal[1]
```

```
##  Accuracy 
## 0.9920497
```

### Analyze results

There's a wide variety of accuracy between the models ranging from ~ 40-99 %. The random forest was the most accurate with a 99.2 % accuracy on a 9811 random out-of-sample test - which equates to an error rate of ~ 0.8 %.  Since the random forest did a better job with all 5 classifiers than any other model, an ensemble stacking approach was not used. However, the Gradient Boosting model can provide some insight on the influence of the variables as show below.

![plot of chunk mod_gbm](figure/mod_gbm-1.png)

```
##                                       var     rel.inf
## roll_belt                       roll_belt 22.74867260
## pitch_forearm               pitch_forearm 10.78620709
## yaw_belt                         yaw_belt  8.81539794
## magnet_dumbbell_z       magnet_dumbbell_z  7.56788909
## magnet_dumbbell_y       magnet_dumbbell_y  6.18299587
## roll_forearm                 roll_forearm  5.15623842
## magnet_belt_z               magnet_belt_z  4.20979409
## gyros_belt_z                 gyros_belt_z  3.37666050
## accel_forearm_x           accel_forearm_x  3.25797925
## pitch_belt                     pitch_belt  2.58424397
## roll_dumbbell               roll_dumbbell  2.22377403
## magnet_forearm_z         magnet_forearm_z  2.07948945
## gyros_dumbbell_y         gyros_dumbbell_y  2.02052864
## accel_dumbbell_y         accel_dumbbell_y  1.94724104
## accel_forearm_z           accel_forearm_z  1.81385857
## accel_dumbbell_x         accel_dumbbell_x  1.50686429
## magnet_dumbbell_x       magnet_dumbbell_x  1.35150708
## magnet_arm_x                 magnet_arm_x  1.33206818
## roll_arm                         roll_arm  1.14711218
## yaw_arm                           yaw_arm  1.07249547
## magnet_arm_z                 magnet_arm_z  0.99077108
## magnet_belt_x               magnet_belt_x  0.91574029
## magnet_belt_y               magnet_belt_y  0.86307437
## magnet_forearm_x         magnet_forearm_x  0.78563531
## magnet_arm_y                 magnet_arm_y  0.61049218
## accel_dumbbell_z         accel_dumbbell_z  0.57562832
## accel_belt_z                 accel_belt_z  0.50475076
## gyros_arm_y                   gyros_arm_y  0.47203896
## total_accel_dumbbell total_accel_dumbbell  0.43645508
## magnet_forearm_y         magnet_forearm_y  0.42084463
## gyros_belt_y                 gyros_belt_y  0.34257534
## gyros_dumbbell_z         gyros_dumbbell_z  0.33504003
## total_accel_forearm   total_accel_forearm  0.26105274
## accel_forearm_y           accel_forearm_y  0.24947843
## gyros_forearm_z           gyros_forearm_z  0.20924481
## gyros_dumbbell_x         gyros_dumbbell_x  0.20512471
## yaw_forearm                   yaw_forearm  0.16939617
## accel_arm_z                   accel_arm_z  0.15358675
## accel_arm_x                   accel_arm_x  0.14876508
## pitch_dumbbell             pitch_dumbbell  0.12459686
## gyros_forearm_x           gyros_forearm_x  0.04469034
## total_accel_belt         total_accel_belt  0.00000000
## gyros_belt_x                 gyros_belt_x  0.00000000
## accel_belt_x                 accel_belt_x  0.00000000
## accel_belt_y                 accel_belt_y  0.00000000
## pitch_arm                       pitch_arm  0.00000000
## total_accel_arm           total_accel_arm  0.00000000
## gyros_arm_x                   gyros_arm_x  0.00000000
## gyros_arm_z                   gyros_arm_z  0.00000000
## accel_arm_y                   accel_arm_y  0.00000000
## yaw_dumbbell                 yaw_dumbbell  0.00000000
## gyros_forearm_y           gyros_forearm_y  0.00000000
```

### Use model to predict on 20 observation test set


```r
predict(mod_rf, tst)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


