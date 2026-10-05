tune_xgb <- function(task, n_evals = 500, workers = parallelly::availableCores(omit = 1)) {
  learner <- lrn(
    "regr.xgboost",
    early_stopping_rounds = 50,
    eval_metric = "rmse",
    nrounds = to_tune(upper = 5000, internal = TRUE),
    max_depth = to_tune(2, 12), # glex(max_interaction) caps interaction order post hoc
    subsample = to_tune(0.1, 1),
    colsample_bytree = to_tune(0.5, 1),
    eta = to_tune(1e-3, 0.3, logscale = TRUE),
    lambda = to_tune(1e-3, 1e3, logscale = TRUE),
    alpha = to_tune(1e-3, 1e2, logscale = TRUE),
    min_child_weight = to_tune(1, 100, logscale = TRUE),
    gamma = to_tune(1e-3, 1e3, logscale = TRUE)
  )
  set_validate(learner, "test")

  tuned <- auto_tuner(
    # Batch MBO proposes one point at a time; async keeps all workers busy
    tuner = tnr("async_mbo"),
    learner = learner,
    resampling = rsmp("cv", folds = 3),
    measure = msr("regr.mse"),
    terminator = trm("evals", n_evals = n_evals, k = 0),
    store_tuning_instance = TRUE,
    store_benchmark_result = TRUE,
    # Async archive lives in Redis; freeze it so the stored target is self-contained
    callbacks = clbk("mlr3tuning.async_freeze_archive")
  )
  tuned$id <- "xgboost"

  # Requires a running Redis server (REDIS_URL or localhost default)
  mirai::daemons(workers, seed = 2026L)
  on.exit(mirai::daemons(0))
  rush::rush_plan(n_workers = workers, worker_type = "mirai")

  # No $marshal() needed: xgboost v3 boosters serialize natively
  tuned$train(task)
  tuned
}
