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

# xgboost handle sfactors since v3
make_bike_task <- function(bike) {
  bike <- copy(bike)
  bike[, let(
    mnth = NULL,
    workingday = as.integer(workingday == "Workingday"),
    weathersit = factor(gsub("[[:space:]/]", "_", weathersit))
  )]
  as_task_regr(model.matrix(~ -1 + ., bike), target = "bikers", id = "bikeshare_xgb")
}
