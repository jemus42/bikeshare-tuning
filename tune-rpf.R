pak::pak(c("mlr3verse", "mlr-org/mlr3extralearners", "PlantedML/randomPlantedForest"))

library(mlr3verse)
library(mlr3extralearners)

bike <- readRDS("bike.rds")

biketask <- as_task_regr(bike, target = "bikers")
length(biketask$feature_names)

terminator <- trm("evals", n_evals = 500, k = 0)
inner_resampling <- rsmp("cv", folds = 3)
#tuner <- tnr("mbo")
tuner <- tnr("random_search", batch_size = 5)

tuned_rpf <- auto_tuner(
  tuner = tuner,
  learner = lrn("regr.rpf", ntrees = 200, nthreads = 2),
  resampling = inner_resampling,
  terminator = terminator,
  measure = msr("regr.rmse"),
  search_space = ps(
    max_interaction = p_int(2, length(biketask$feature_names)),
    splits = p_int(10, 1000),
    split_try = p_int(1, 20),
    t_try = p_dbl(0.1, 1)
  ),
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE
)

future::plan("multisession", workers = 10)

tuned_rpf$train(biketask)
tuned_rpf$tuning_instance$result
saveRDS(tuned_rpf, "tuned_rpf.rds")
