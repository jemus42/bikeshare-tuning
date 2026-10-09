# Tuning picks the minimum of many noisy 3-fold CV estimates; re-score the top configs on
# fresh repeated CV splits shared by all configs, so they can be compared pairwise
reevaluate_top <- function(tuned, archive, task, model, n_top = 20, folds = 5, repeats = 10, workers = 1) {
  # Final learner carries every fixed param (and for xgboost: no early stopping)
  template <- tuned$unmarshal()$learner$clone(deep = TRUE)$reset()
  top <- archive[state == "finished"][order(regr.mse)][seq_len(n_top)]

  learners <- lapply(seq_len(n_top), function(i) {
    learner <- template$clone(deep = TRUE)
    vals <- top$x_domain[[i]]
    # xgboost's nrounds was set by early stopping, so it is not part of x_domain
    if ("internal_tuned_values_nrounds" %in% names(top)) {
      vals$nrounds <- top$internal_tuned_values_nrounds[i]
    }
    learner$param_set$set_values(.values = vals)
    learner$id <- sprintf("%s_%02i", model, i)
    learner
  })

  resampling <- rsmp("repeated_cv", folds = folds, repeats = repeats)
  # Same seed for every model: tasks share row ids, so all models get identical splits
  set.seed(2026L)
  resampling$instantiate(task)

  mirai::daemons(workers, seed = 2026L)
  on.exit(mirai::daemons(0))
  bmr <- benchmark(benchmark_grid(task, learners, resampling), store_models = FALSE)

  scores <- bmr$score(msr("regr.mse"))[, .(learner_id, iteration, regr.mse)]
  scores[, `:=`(
    model = model,
    rank_tuning = match(learner_id, vapply(learners, \(l) l$id, character(1))),
    learner_id = NULL
  )]
  scores[, tuning_mse := top$regr.mse[rank_tuning]]

  configs <- data.table(
    model = model,
    rank_tuning = seq_len(n_top),
    param_vals = lapply(learners, \(l) l$param_set$values)
  )
  list(scores = scores[], configs = configs)
}

# Lowest mean re-evaluation MSE; iterations of repeated CV are correlated, so the SE is naive
pick_best <- function(reeval) {
  summary <- reeval$scores[,
    .(regr.mse = mean(regr.mse), regr.mse_se = sd(regr.mse) / sqrt(.N), tuning_mse = tuning_mse[1]),
    by = rank_tuning
  ]
  best <- summary[which.min(regr.mse)]
  list(
    param_vals = reeval$configs$param_vals[[best$rank_tuning]],
    rank_tuning = best$rank_tuning,
    regr.mse = best$regr.mse,
    regr.mse_se = best$regr.mse_se,
    tuning_mse = best$tuning_mse
  )
}
