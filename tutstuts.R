

library(tidyverse)
library(parsnip)
library(rsample)
library(dplyr)
library(recipes)
library(themis)
library(tune)
library(yardstick)
library(dplyr)
# Pegar os dados do Aruã --------------------------------------------------

df<-read.csv('BLUSOL/BLUSOL/df.csv')


# Modelo classificação ----------------------------------------------------


# SMOTE -------------------------------------------------------------------

df$DEFAULT<-as.factor(df$DEFAULT)
df1 <- themis::smote(df,var = "DEFAULT", k = 6, over_ratio = 1)
prop.table(table(df1$DEFAULT))



# Split -------------------------------------------------------------------


split <- initial_split(df1, prop = 0.75)
x_train <- training(split)
x_test <- testing(split)


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
                      metrics = metric_set(accuracy,kap))


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



# Random forest -----------------------------------------------------------

rand_forest_ranger_spec <-
  rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine('ranger') %>%
  set_mode('classification')

doParallel::registerDoParallel()

boost_grid<-tune_grid(rand_forest_ranger_spec, 
                      receita2,
                      resamples = cv_split,
                      grid = 10,
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
  predict(new_data = tst_proc) %>%
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

boost_fit2 <- finalize_model(svm_linear_kernlab_spec, parameters = best) %>% 
  fit(DEFAULT~.,tr_proc)
boost_fit2

saveRDS(boost_fit2,"SVM.rda")




fitted<-boost_fit2 %>%
  predict(new_data = tst_proc) %>%
  mutate(observado = tst_proc$DEFAULT,
         modelo = "SVM Tuned")

xc<-rbind(xc,fitted)

resultado <- xc %>% 
  group_by(modelo) %>%
  dplyr::filter(modelo=="SVM Tuned") %>% 
  metrics(truth = observado, estimate = .pred_class)

resultado
