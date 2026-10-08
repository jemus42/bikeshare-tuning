# Small, committable outputs; the full AutoTuner objects in _targets/ are >300MB each

export_archive <- function(archive, path) {
  archive <- archive[state == "finished"]
  # x_domain holds values on the learner's scale, not the logscale search space
  out <- cbind(
    rbindlist(archive$x_domain),
    archive[, .SD, .SDcols = patterns("^regr\\.mse$|^internal_tuned_values_|^runtime_learners$")]
  )
  setorder(out, regr.mse)
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
