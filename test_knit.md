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

There's a wide variety of accuracy between the models ranging from ~ 40-99 %. The random forest was the most accurate with the ~ 1% error rate.  Since the random forest did a better job with all 5 classifiers than any other model, an ensemble stacking approach was not used. However, the Gradient Boosting model can provide some insight on the influence of the variables as show below.


```
##                   Length Class      Mode     
## initF                 1  -none-     numeric  
## fit               49055  -none-     numeric  
## train.error         150  -none-     numeric  
## valid.error         150  -none-     numeric  
## oobag.improve       150  -none-     numeric  
## trees               750  -none-     list     
## c.splits              0  -none-     list     
## bag.fraction          1  -none-     numeric  
## distribution          1  -none-     list     
## interaction.depth     1  -none-     numeric  
## n.minobsinnode        1  -none-     numeric  
## num.classes           1  -none-     numeric  
## n.trees               1  -none-     numeric  
## nTrain                1  -none-     numeric  
## train.fraction        1  -none-     numeric  
## response.name         1  -none-     character
## shrinkage             1  -none-     numeric  
## var.levels           52  -none-     list     
## var.monotone         52  -none-     numeric  
## var.names            52  -none-     character
## var.type             52  -none-     numeric  
## verbose               1  -none-     logical  
## classes               5  -none-     character
## estimator         49055  -none-     numeric  
## data                  6  -none-     list     
## xNames               52  -none-     character
## problemType           1  -none-     character
## tuneValue             4  data.frame list     
## obsLevels             5  -none-     character
## param                 1  -none-     list
```

### Use model to predict on 20 observation test set


```r
predict(mod_rf, tst)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


