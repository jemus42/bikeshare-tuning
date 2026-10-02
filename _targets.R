library(targets)

tar_option_set(
  packages = c("data.table", "mlr3", "mlr3learners", "mlr3tuning", "mlr3mbo")
)

tar_source()

list(
  tar_target(bike_raw, load_bikeshare()),
  tar_target(bike, preprocess_bike(bike_raw)),
  tar_target(task_rpf, as_task_regr(bike, target = "bikers", id = "bikeshare")),
  tar_target(task_xgb, make_task_xgb(bike)),
  tar_target(tuned_xgb, tune_xgb(task_xgb, n_evals = 1000)),
  tar_target(result_xgb, tuned_xgb$tuning_instance$result),
  tar_target(archive_xgb, as.data.table(tuned_xgb$archive))
)
