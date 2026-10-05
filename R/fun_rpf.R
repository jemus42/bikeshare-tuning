tune_rpf <- function(task, n_evals = 500, workers = parallelly::availableCores(omit = 1)) {
  learner <- lrn(
    "regr.rpf",
    ntrees = 200
    max_interaction = to_tune(2, length(biketask$feature_names)),
    splits = to_tune(10, 500),
    split_try = to_tune(1, 50),
    t_try = p_dbl(0.01, 1)
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
  tuned$id <- "rpf"

  # Requires a running Redis server (REDIS_URL or localhost default)
  mirai::daemons(workers, seed = 2026L)
  on.exit(mirai::daemons(0))
  rush::rush_plan(n_workers = workers, worker_type = "mirai")

  tuned$train(task)
  # $marshal() needed for serialization
  tuned$marshal()
  tuned
}
