library(tidyverse)
library(parsnip)
library(rsample)
library(dplyr)
library(recipes)
library(themis)
library(tune)
library(yardstick)
library(dplyr)
library(pROC)

# Arquivo Feather ---------------------------------------------------------
library(arrow)
library(feather)

df<-arrow::read_feather("BLUSOL/BLUSOL/big_frame.feather")

# Pegar os dados do Aruã --------------------------------------------------

# 
# 
# df<-read.csv('BLUSOL/BLUSOL/df.csv')


# Modelo classificação ----------------------------------------------------


# SMOTE -------------------------------------------------------------------
df$classificacao<- NULL
df$DEFAULT<-as.factor(df$DEFAULT)
# df1 <- themis::smote(df,var = "DEFAULT", k = 6, over_ratio = 1)
# prop.table(table(df1$DEFAULT))



# Split -------------------------------------------------------------------


split <- initial_split(df1, prop = 0.65)
x_train <- training(split)
x_test <- testing(split)
x_train <- themis::smote(x_train,var = "DEFAULT", k = 5, over_ratio = 1)

## Recipes


recipe1 <- recipes::recipe(DEFAULT~.,x_train) %>% 
  step_unknown(0) %>% 
  prep()


receita_prep <- prep(recipe1)

tr_proc <- bake(receita_prep, new_data = NULL)
tst_proc <- bake(receita_prep, new_data = x_test)


#Prop teste
prop.table(table(tst_proc$DEFAULT))


#Prop treino
prop.table(table(tr_proc$DEFAULT))


#### K folds validação cruzada

receita2 <- recipe(DEFAULT~.,tr_proc) %>% prep()


cv_split<-vfold_cv(tr_proc, v=5)



### Modelos de classificação

### XGBoost 

boost_tree_xgboost_spec <-
  boost_tree(tree_depth = tune(), trees = tune(), learn_rate = tune(), min_n = tune(), loss_reduction = tune(), sample_size = tune(), stop_iter = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('classification')


### XGBoost Ajuste de hiper parâmetros

doParallel::registerDoParallel()

boost_grid<-tune_grid(boost_tree_xgboost_spec, 
                      receita2,
                      resamples = cv_split,
                      grid = 15,
                      metrics = metric_set(yardstick::roc_auc,accuracy,kap))
 

#### Métricas XGBoost

boost_grid %>% 
  collect_metrics() %>% 
  head()
best<-boost_grid %>% 
  select_best("kap")



### Finalizando Boost

boost_fit <- finalize_model(boost_tree_xgboost_spec, parameters = best) %>% 
  fit(DEFAULT~.,tr_proc)
boost_fit

saveRDS(boost_fit,"xgboost.rda")


xc<-boost_fit %>% 
  predict(new_data = tst_proc) %>% 
  mutate(observado = tst_proc$DEFAULT,
         modelo = "XGBoost Tuned")

resultado_xg <- xc %>% 
  group_by(modelo) %>% 
  metrics(truth = observado, estimate = .pred_class)

resultado_xg



# Curva AUC XGBoost ---------------------------------------------------------------


xg_resultado <- boost_fit %>% predict(new_data = tst_proc, type = "prob") %>% 
  mutate(DEFAULT = tst_proc$DEFAULT)

yardstick::roc_auc(xg_resultado, DEFAULT, .pred_True)



# Plot tune results -------------------------------------------------------

boost_grid<-tune_grid(boost_tree_xgboost_spec, 
                      receita2,
                      resamples = cv_split,
                      grid = 15,
                      metrics = metric_set(yardstick::roc_auc))

autoplot(boost_grid)

# Random forest -----------------------------------------------------------

rand_forest_ranger_spec <-
  rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine('ranger') %>%
  set_mode('classification')

doParallel::registerDoParallel()

boost_grid<-tune_grid(rand_forest_ranger_spec, 
                      receita2,
                      resamples = cv_split,
                      grid = 15,
                      metrics = metric_set(accuracy,kap))

boost_grid %>% 
  collect_metrics() %>% 
  head()
best<-boost_grid %>% 
  select_best("kap")

boost_fit2 <- finalize_model(rand_forest_ranger_spec, parameters = best) %>% 
  fit(DEFAULT~.,tr_proc)
boost_fit2

saveRDS(boost_fit2,"randomforest.rda")

fitted<-boost_fit2 %>%
  predict(new_data = tst_proc, type = "prob") %>%
  mutate(observado = tst_proc$DEFAULT,
         modelo = "Random Tuned")

xc<-rbind(xc,fitted)

resultado_fin <- xc %>% 
  group_by(modelo) %>%
  dplyr::filter(modelo=="Random Tuned") %>% 
  metrics(truth = observado, estimate = .pred_class)

resultado_fin



# SVM ---------------------------------------------------------------------

svm_linear_kernlab_spec <-
  svm_linear(cost = tune(), margin = tune()) %>%
  set_engine('kernlab') %>%
  set_mode('classification')


boost_grid<-tune_grid(svm_linear_kernlab_spec, 
                      receita2,
                      resamples = cv_split,
                      grid = 10,
                      metrics = metric_set(accuracy,kap))

boost_grid %>% 
  collect_metrics() %>% 
  head()
best<-boost_grid %>% 
  select_best("kap")

boost_fit3 <- finalize_model(svm_linear_kernlab_spec, parameters = best) %>% 
  fit(DEFAULT~.,tr_proc)
boost_fit3

saveRDS(boost_fit3,"SVM.rda")




fitted<-boost_fit3 %>%
  predict(new_data = tst_proc) %>%
  mutate(observado = tst_proc$DEFAULT,
         modelo = "SVM Tuned")

xc<-rbind(xc,fitted)

resultado <- xc %>% 
  group_by(modelo) %>%
  dplyr::filter(modelo=="SVM Tuned") %>% 
  metrics(truth = observado, estimate = .pred_class)

resultado




# Testando vetores de saída com xgboost e random forest -------------------


boost_fit$censor_probs

prob<-predict(boost_fit, tst_proc, type = "prob")




prob<-cbind(prob,tst_proc)
prob<-prob %>% select(c(.pred_FALSE,.pred_TRUE,DEFAULT))


# Conta de padeiro do Aruã

# ((prob %>% dplyr::filter(DEFAULT=="True"))*14+
#   (1-(prob %>%dplyr::filter(DEFAULT=="False") %>% count())))/
#   ((prob %>% dplyr::filter(DEFAULT== "True") %>% count())*14+
#   (prob %>% dplyr::filter(DEFAULT== "False") %>% count()))


x<-prob %>% select(c(DEFAULT,.pred_TRUE)) %>% dplyr::filter(DEFAULT=="TRUE")
y<-prob %>% select(c(DEFAULT,.pred_FALSE)) %>% dplyr::filter(DEFAULT=="FALSE")
h<-prob %>% select(c(DEFAULT,.pred_FALSE)) %>% dplyr::filter(DEFAULT=="TRUE")
z<-prob %>% select(c(DEFAULT,.pred_TRUE)) %>% dplyr::filter(DEFAULT=="FALSE")

(sum(x$.pred_TRUE)*14+(sum(y$.pred_FALSE)))/((x %>% select(.pred_TRUE) %>% count())*14+
  (y %>% select(.pred_FALSE) %>% count()))


# Caso queira transformar em binario --------------------------------------



prob<-predict(boost_fit2, tst_proc, type = "prob")
prob<-cbind(prob,tst_proc)
prob<-prob %>% select(c(.pred_FALSE,.pred_TRUE,DEFAULT))


# Conta de padeiro do Aruã


x<-prob %>% select(c(DEFAULT,.pred_TRUE)) %>% dplyr::filter(DEFAULT=="TRUE")
y<-prob %>% select(c(DEFAULT,.pred_FALSE)) %>% dplyr::filter(DEFAULT=="FALSE")
h<-prob %>% select(c(DEFAULT,.pred_FALSE)) %>% dplyr::filter(DEFAULT=="TRUE")
z<-prob %>% select(c(DEFAULT,.pred_TRUE)) %>% dplyr::filter(DEFAULT=="FALSE")

(sum(x$.pred_TRUE)*14+(sum(y$.pred_FALSE)))/((x %>% select(.pred_TRUE) %>% count())*14+
                                               (y %>% select(.pred_FALSE) %>% count()))


# Curva ROC ---------------------------------------------------------------
xc$observado<-as.integer(as.logical(xc$observado))
xc$.pred_class<- as.integer(as.logical(xc$.pred_class))
# Threshold <- prob %>% roc_auc(truth = DEFAULT, .pred_FALSE)


roc_ <- roc(xc$observado, xc$.pred_class, smoothed = TRUE, plot = TRUE)

plot(roc_svm_test, add = TRUE, col = "red", print.auc = TRUE, print.auc.x = 0.5, print.auc.y = 0.3)
legend(0.3, 0.2, legend = c("test-svm"), lty = c(1), col = c("blue"))



