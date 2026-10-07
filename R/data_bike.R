load_bikeshare <- function() {
  data("Bikeshare", package = "ISLR2", envir = environment())
  as.data.table(Bikeshare)
}

preprocess_bike <- function(bike_raw) {
  bike <- copy(bike_raw)
  bike[, let(
    hr = as.numeric(as.character(hr)),
    workingday = factor(
      workingday,
      levels = c(0, 1),
      labels = c("No Workingday", "Workingday")
    ),
    season = factor(season, levels = 1:4, labels = c("Winter", "Spring", "Summer", "Fall")),
    # atemp is near-collinear with temp; registered + casual == bikers
    atemp = NULL,
    day = NULL,
    registered = NULL,
    casual = NULL
  )]
  bike[]
}

# rpf and xgboost (v3) both handle factors natively
make_bike_task <- function(bike, encode = FALSE) {
  bike <- copy(bike)
  bike[, let(
    mnth = NULL,
    weathersit = factor(gsub("[[:space:]/]", "_", weathersit))
  )]
  # ponytail: mlr3learners' regr.xgboost rejects factor features, drop once it accepts them
  if (encode) {
    bike[, workingday := as.integer(workingday == "Workingday")]
    return(as_task_regr(model.matrix(~ -1 + ., bike), target = "bikers", id = "bikeshare_encoded"))
  }
  as_task_regr(bike, target = "bikers", id = "bikeshare")
}
