library(targets)

tar_option_set(
  packages = c("data.table", "mlr3", "mlr3learners", "mlr3tuning", "mlr3mbo")
)

tar_source()

tuning_n_evals <- 1000
tuning_workers <- floor(parallelly::availableCores(omit = 1, which = "max") / 2)

list(
  tar_target(bike_raw, load_bikeshare()),
  tar_target(bike, preprocess_bike(bike_raw)),
  tar_target(bike_task, make_bike_task(bike)),
  tar_target(bike_task_encoded, make_bike_task(bike, encode = TRUE)),
  # xgboost ----
  tar_target(
    tuned_xgb,
    tune_xgb(bike_task_encoded, n_evals = tuning_n_evals, workers = tuning_workers)
  ),
  tar_target(result_xgb, tuned_xgb$tuning_result),
  tar_target(archive_xgb, as.data.table(tuned_xgb$archive)),
  # rpf ----
  tar_target(
    tuned_rpf,
    tune_rpf(bike_task, n_evals = tuning_n_evals, workers = tuning_workers),
    packages = "mlr3extralearners"
  ),
  tar_target(result_rpf, tuned_rpf$unmarshal()$tuning_result),
  tar_target(archive_rpf, as.data.table(tuned_rpf$unmarshal()$archive)),
  # re-evaluation of top configs ----
  tar_target(
    reeval_xgb,
    reevaluate_top(tuned_xgb, archive_xgb, bike_task_encoded, model = "xgb", workers = tuning_workers)
  ),
  tar_target(
    reeval_rpf,
    reevaluate_top(tuned_rpf, archive_rpf, bike_task, model = "rpf", workers = tuning_workers),
    packages = "mlr3extralearners"
  ),
  # exports (committed) ----
  tar_target(xgb_archive_file, export_archive(archive_xgb, "results/xgb_archive.csv"), format = "file"),
  tar_target(rpf_archive_file, export_archive(archive_rpf, "results/rpf_archive.csv"), format = "file"),
  tar_target(
    params_file,
    export_params("results/best_params.json", xgboost = reeval_xgb, rpf = reeval_rpf),
    format = "file"
  ),
  tar_target(reeval_file, export_reeval("results/reeval.csv", reeval_xgb, reeval_rpf), format = "file"),
  tar_target(
    search_space_file,
    export_search_space("results/search_space.csv", xgb = tuned_xgb, rpf = tuned_rpf),
    format = "file",
    packages = "mlr3extralearners"
  )
)
