

library(tidyverse)
library(parsnip)





# Pegar os dados do Aruã --------------------------------------------------




# Modelo classificação ----------------------------------------------------



# Split -------------------------------------------------------------------


split <- initial_split(f, prop = 0.75)
x_train <- training(split)
x_test <- testing(split)


## Recipes


recipe1 <- recipes::recipe(clusters~.,x_train) %>% 
  step_rm (c("NOME_DO_PRODUTO","CNPJ_CLIENTE_PJ")) %>%
  step_unknown(0) %>% 
  step_dummy(c("PORTE_CLIENTE_PJ","classification"),one_hot = TRUE) %>% 
  prep()


receita_prep <- prep(recipe1)

tr_proc <- bake(receita_prep, new_data = NULL)
tst_proc <- bake(receita_prep, new_data = x_test)


#Prop teste
prop.table(table(tst_proc$clusters))


#Prop treino
prop.table(table(tr_proc$clusters))

#### K folds validação cruzada

receita2 <- recipe(clusters~.,tr_proc) %>% prep()


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
  select_best("accuracy")



### Finalizando Boost

boost_fit <- finalize_model(boost_tree_xgboost_spec, parameters = best) %>% 
  fit(clusters~.,tr_proc)
boost_fit

saveRDS(boost_fit,"xgboost.rda")


xc<-boost_fit %>% 
  predict(new_data = tst_proc) %>% 
  mutate(observado = tst_proc$clusters,
         modelo = "XGBoost Tuned")

resultado_xg <- fitted %>% 
  group_by(modelo) %>% 
  metrics(truth = observado, estimate = .pred_class)

resultado_xg

