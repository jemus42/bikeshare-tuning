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
  out <- lapply(list(...), function(reeval) {
    best <- pick_best(reeval)
    # JSON has no Inf; the only one is rpf's max_interaction_limit init value
    best$param_vals <- Filter(function(v) !(is.numeric(v) && !is.finite(v)), best$param_vals)
    best
  })
  jsonlite::write_json(out, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  path
}

export_reeval <- function(path, ...) {
  fwrite(rbindlist(lapply(list(...), \(reeval) reeval$scores)), path)
  path
}

# Bounds on the tuning scale (log for logscale params), so the report matches whichever run produced the archives
export_search_space <- function(path, ...) {
  tuned <- list(...)
  out <- rbindlist(lapply(names(tuned), function(model) {
    search_space <- tuned[[model]]$unmarshal()$tuning_instance$search_space
    data.table(
      model = model,
      param = search_space$ids(),
      lower = search_space$lower,
      upper = search_space$upper,
      logscale = search_space$is_logscale
    )
  }))
  fwrite(out[!is.na(lower)], path)
  path
}
