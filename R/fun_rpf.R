tune_rpf <- function(task, n_evals = 500, workers = parallelly::availableCores(omit = 1)) {
  learner <- lrn(
    "regr.rpf",
    ntrees = 200,
    # 1 = additive model; upper end lets trees use every feature
    max_interaction = to_tune(1, length(task$feature_names)),
    splits = to_tune(p_int(10, 1000, logscale = TRUE)),
    split_structure = to_tune(),
    split_try = to_tune(p_int(1, 50, logscale = TRUE)),
    # Below 1 / possible splits, ceil() leaves a single candidate, so tiny values are harmless
    t_try = to_tune(p_dbl(1e-4, 1, logscale = TRUE)),
    # caps t_try: candidates per round = min(t_try * possible splits, max_candidates)
    max_candidates = to_tune(p_int(10, 500, logscale = TRUE)),
    # 0 = uniform candidate sampling
    split_decay_rate = to_tune(0, 1),
    delete_leaves = to_tune()
  )

  tuned <- auto_tuner(
    # Batch MBO proposes one point at a time; async keeps all workers busy
    tuner = tnr("async_mbo"),
    learner = learner,
    resampling = rsmp("cv", folds = 3),
    measure = msr("regr.mse"),
    terminator = trm_evals_or_stagnation(n_evals),
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
