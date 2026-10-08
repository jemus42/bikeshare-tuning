# Small, committable outputs; the full AutoTuner objects in _targets/ are >300MB each

export_archive <- function(archive, path) {
  archive <- archive[state == "finished"]
  setorder(archive, timestamp_ys)
  # Folds are instantiated once per tuning run, so fold scores are paired across configs
  folds <- rbindlist(lapply(archive$resample_result, function(rr) {
    as.list(setNames(rr$score(msr("regr.mse"))$regr.mse, paste0("mse_fold", seq_len(rr$iters))))
  }))
  # x_domain holds values on the learner's scale, not the logscale search space
  out <- cbind(
    eval = seq_len(nrow(archive)),
    # acq_cb is NA for the initial design, which MBO evaluates before fitting a surrogate
    init_design = is.na(archive$acq_cb),
    rbindlist(archive$x_domain),
    archive[, .SD, .SDcols = patterns("^regr\\.mse$|^internal_tuned_values_|^runtime_learners$")],
    folds
  )
  fwrite(out, path)
  path
}

export_params <- function(path, ...) {
  results <- list(...)
  out <- lapply(results, function(r) {
    vals <- r$learner_param_vals[[1]]
    # JSON has no Inf; the only one is rpf's max_interaction_limit init value
    vals <- Filter(function(v) !(is.numeric(v) && !is.finite(v)), vals)
    list(param_vals = vals, regr.mse = r$regr.mse)
  })
  jsonlite::write_json(out, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  path
}
