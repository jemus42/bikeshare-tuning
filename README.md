# bikeshare-tuning
Tuning rpf and xgb on Bikeshare for future reference

Run with `targets::tar_make()`; load results via `targets::tar_read(result_xgb)` etc.

## Results

Tuned hyperparameters (3-fold CV MSE, 1000 async MBO evals each) are committed in `results/`:

- `best_params.json`: full `param_vals` for the best xgboost and rpf configs
- `xgb_archive.csv`, `rpf_archive.csv`: all evaluated configs on the learner scale in evaluation order, with per-fold MSE
- `search_space.csv`: tuning bounds of the run that produced the archives (log scale where `logscale` is TRUE)

`tuning-report.qmd` analyzes the archives (convergence, CV noise, search space bounds); render with `make report`.

Refit without tuning, using `make_bike_task()` from `R/data_bike.R`. xgboost (`mlr3learners`) needs `encode = TRUE`; rpf (`mlr3extralearners`) takes ~3 min single-threaded, set `nthreads` to speed up:

```r
params <- jsonlite::read_json("results/best_params.json", simplifyVector = TRUE)
learner <- lrn("regr.rpf")
learner$param_set$set_values(.values = params$rpf$param_vals)
learner$train(make_bike_task(preprocess_bike(load_bikeshare())))
```
