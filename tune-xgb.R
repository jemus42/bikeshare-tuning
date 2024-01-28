pak::pak(c("xgboost", "mlr3verse", "mlr3mbo", "mlr3tuning"))

library(mlr3verse)
library(mlr3extralearners)

bike <- readRDS("bike.rds")

biketask <- as_task_regr(bike, target = "bikers")
length(biketask$feature_names)

# terminator <- trm("evals", n_evals = 200, k = 20)
terminator <- trm("evals", n_evals = 50, k = 0)

inner_resampling <- rsmp("cv", folds = 3)
tuner <- tnr("mbo")

learner = po("encode", method = "one-hot") %>>%
  po("learner",
     lrn("regr.xgboost",
         early_stopping_rounds = 100,
         early_stopping_set = "test",
         eval_metric = "error",
         nrounds = 5000)
  ) |>
  as_learner()

tuned_xgb <- auto_tuner(
  tuner = tuner,
  learner = learner,
  resampling = inner_resampling,
  measure = msr("regr.rmse"),
  terminator = terminator,
  search_space = ps(
    regr.xgboost.max_depth = p_int(2, length(biketask$feature_names)),
    regr.xgboost.subsample = p_dbl(0.1, 1),
    regr.xgboost.colsample_bytree = p_dbl(0.1, 1),
    #regr.xgboost.nrounds = p_int(10, 5000),
    regr.xgboost.eta = p_dbl(1e-4, 1, logscale = TRUE),
    regr.xgboost.lambda = p_dbl(1e-3, 1, logscale = TRUE),
    regr.xgboost.alpha = p_dbl(1e-3, 1, logscale = TRUE)
  ),
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE
)
tuned_xgb$id <- "xgboost"

future::plan("multisession")

tuned_xgb$train(biketask)
tuned_xgb$tuning_instance$result
tuned_xgb$archive
saveRDS(tuned_xgb, "tuned_xgb.rds")
