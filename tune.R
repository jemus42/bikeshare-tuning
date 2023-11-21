pak::pak(c("xgboost", "mlr3verse", "mlr-org/mlr3extralearners", "PlantedML/randomPlantedForest"))

library(mlr3verse)
library(mlr3extralearners)

bike <- readRDS("bike.rds")

biketask <- as_task_regr(bike, target = "bikers")
length(biketask$feature_names)
splits <- partition(biketask)

terminator <- trm("evals", n_evals = 100, k = 10)
inner_resampling <- rsmp("cv", folds = 3)
tuner <- tnr("mbo")


tuned_rpf <- auto_tuner(
  tuner = tuner,
  learner = lrn("regr.rpf", ntrees = 200),
  resampling = inner_resampling,
  terminator = terminator,
  search_space = ps(
    max_interaction = p_int(2, length(biketask$feature_names)),
    splits = p_int(10, 100),
    split_try = p_int(1, 20),
    t_try = p_dbl(0.1, 1)
  ),
  store_tuning_instance = TRUE, store_benchmark_result = TRUE
)

tuned_xgb <- auto_tuner(
  tuner = tuner,
  learner = po("encode", method = "one-hot") %>>%
    po("learner", lrn("regr.xgboost")) |>
    as_learner(id = "xgboost"),
  resampling = inner_resampling,
  terminator = terminator,
  search_space = ps(
    regr.xgboost.max_depth = p_int(2, length(biketask$feature_names)),
    regr.xgboost.subsample = p_dbl(0.1, 1),
    regr.xgboost.colsample_bytree = p_dbl(0.1, 1),
    regr.xgboost.nrounds = p_int(10, 5000),
    regr.xgboost.eta = p_dbl(0, 1)
  ),
  store_tuning_instance = TRUE, store_benchmark_result = TRUE
)

future::plan("multisession")

tuned_xgb$train(biketask, row_ids = splits$train)
tuned_xgb$tuning_instance$result
saveRDS(tuned_xgb, "tuned_xgb.rds")

tuned_rpf$train(biketask, row_ids = splits$train)
tuned_rpf$tuning_instance$result
saveRDS(tuned_rpf, "tuned_rpf.rds")


autoplot(tuned_rpf$tuning_instance)
autoplot(tuned_xgb$tuning_instance)


