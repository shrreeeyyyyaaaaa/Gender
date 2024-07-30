# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

system("ls ../input")

# Any results you write to the current directory are saved as output.
require(dplyr)
require(tidyr)
require(caret)
require(corrplot)
require(Hmisc)
require(ggthemes)
### load in original dataset
voice_Original <- read_csv("../input/voice.csv",col_names = TRUE)
### visual exploration of the dataset for correlation
predictor_Corr <- cor(voice_Original[,-21])
corrplot(predictor_Corr,method="number")

### pre-process of original dataset to remove correlations
pca_Transform <- preProcess(voice_Original,method=c("scale","center","pca"))
voice_Original <- predict(pca_Transform,voice_Original)


### split original dataset into training and testing subsets
sample_Index <- createDataPartition(voice_Original$label,p=0.7,list=FALSE)
voice_Train <- voice_Original[sample_Index,]
voice_Test <- voice_Original[-sample_Index,]

### visual explorations
# correlation plot
new_Corr <- cor(voice_Original[,2:11])
corrplot(new_Corr)

voice_Original%>%
  ggplot(aes(x=PC1,y=PC2))+
  geom_point(aes(color=label))+
  theme_wsj()

voice_Original%>%
  ggplot(aes(x=PC2,y=PC3))+
  geom_point(aes(color=label))+
  theme_wsj()

# set formula
model_Formula <- label~PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10

### set tuning and cross validation paramters
modelControl <- trainControl(method="repeatedcv",number=5,repeats=10)

## model 1: logistic regression
glm_Model <- train(
  model_Formula,
  data=voice_Train,
  method="glm",
  trControl=modelControl
)

## prediction with glm model
voice_Test$glmPrediction <- predict(glm_Model,newdata=voice_Test[,2:11])
## view prediction results
glm_Model   ### accuracy 0.9709 kappa 0.9418
table(voice_Test$label,voice_Test$glmPrediction)

## model 2: random forest
rf_Model <- train(
  model_Formula,
  data=voice_Train,
  method="rf",
  ntrees=1000,
  trControl=modelControl
)
voice_Test$rfPrediction <- predict(rf_Model,newdata=voice_Test[,2:11])
## view model performance
rf_Model  ## accuracy 0.967 kappa 0.934
table(voice_Test$label,voice_Test$rfPrediction)

## model 3: support vector machine
svm_Model <- train(
  model_Formula,
  data=voice_Train,
  method="svmRadial",
  trControl=modelControl
)
## view model performance
svm_Model  ## accuracy 0.974 kappa 0.949
voice_Test$svmPrediction <- predict(svm_Model,newdata=voice_Test[,2:11])
table(voice_Test$label,voice_Test$svmPrediction)

## mdoel 4: gradient boosting machine
gbm_Model <- train(
  model_Formula,
  data=voice_Train,
  method="gbm",
  trControl=modelControl
)
## view model performance
gbm_Model  ## best performance @ accuracy 0.968 kappa 0.935
voice_Test$gbmPrediction <- predict(gbm_Model,newdata=voice_Test[,2:11])
table(voice_Test$label,voice_Test$gbmPrediction)


### compare model performance of 4 models that have been built
model_Comparison <- resamples(
  list(
    LogisticRegression=glm_Model,
    RandomForest=rf_Model,
    SupportVectorMachine=svm_Model,
    GradientBoosting=gbm_Model
  )
)

summary(model_Comparison)

## visual comparison of model performances
bwplot(model_Comparison,layout=c(2,1))
