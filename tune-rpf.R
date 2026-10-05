pak::pak(c("mlr3tuning", "mlr-org/mlr3extralearners", "PlantedML/randomPlantedForest"))

library(mlr3tuning)
library(mlr3extralearners)

bike <- readRDS("bike.rds")

biketask <- as_task_regr(bike, target = "bikers")
length(biketask$feature_names)

terminator <- trm("evals", n_evals = 20, k = 0)
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
tuned_rpf$marshal()
saveRDS(tuned_rpf, "tuned_rpf.rds")

reloaded_rpf <- readRDS("tuned_rpf.rds")
reloaded_rpf$unmarshal()
reloaded_rpf$tuning_instance$result
reloaded_rpf$predict(biketask, row_ids = 1:10)

system.time({
  reloaded_rpf$predict(biketask)
})
