pak::pak(c("xgboost", "mlr3verse", "mlr3mbo", "mlr3tuning"))
pak::pak(c("DiceKriging", "rgenoud"))

library(mlr3verse)
library(mlr3extralearners)

n_cores <- ceiling(parallelly::availableCores() / 2)

bike <- readRDS("bike.rds")

bike[, let(
  mnth = NULL,
  workingday = as.integer(workingday == "Workingday"),
  weathersit = factor(stringr::str_replace_all(
    weathersit,
    "[\\s\\/]",
    replacement = "_"
  ))
)]

# bike$weathersit |> levels()

bike <- model.matrix(~ -1 + ., bike)

biketask <- as_task_regr(bike, target = "bikers")
length(biketask$feature_names)

terminator <- trm("evals", n_evals = 500, k = 0)

inner_resampling <- rsmp("cv", folds = 3)
tuner <- tnr("mbo")

lrn_xgb <- lrn(
  "regr.xgboost",
  early_stopping_rounds = 50,
  eval_metric = "rmse",
  nrounds = to_tune(upper = 5000, internal = TRUE),
  max_depth = to_tune(2, 8), # Not super deep to keep glex() functional decomp feasible
  subsample = to_tune(0.1, 1),
  colsample_bytree = to_tune(0.1, 1),
  eta = to_tune(1e-4, 1, logscale = TRUE),
  lambda = to_tune(1e-3, 1, logscale = TRUE),
  alpha = to_tune(1e-3, 1, logscale = TRUE)
)

learner <- lrn_xgb
set_validate(learner, "test")

tuned_xgb <- auto_tuner(
  tuner = tuner,
  learner = learner,
  resampling = inner_resampling,
  measure = msr("regr.mse"),
  terminator = terminator,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE
)
tuned_xgb$id <- "xgboost"

mirai::daemons(4, .compute = "mlr3_parallelization", seed = 2026L)

tuned_xgb$train(biketask)
tuned_xgb$predict(biketask)$score(msrs(c("regr.rmse", "regr.rmsle")))


saveRDS(tuned_xgb$tuning_instance$result, "result-xgb.rds")
saveRDS(as.data.table(tuned_xgb$archive), "archive-xgb.rds")
tuned_xgb$marshal()
saveRDS(tuned_xgb, "tuned_xgb.rds")

mirai::daemons(0)
