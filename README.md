# bikeshare-tuning
Tuning rpf and xgb on Bikeshare for future reference

Run with `targets::tar_make()`; load results via `targets::tar_read(result_xgb)` etc.

## Results

Tuned with 1000 async MBO evals each (3-fold CV MSE); the top 20 configs per model were then re-evaluated with 10 × 5-fold CV on shared splits. Committed in `results/`:

- `best_params.json`: full `param_vals` of the re-evaluation winners, with their re-evaluation MSE (`regr.mse`, naive `regr.mse_se`) and tuning rank and MSE
- `reeval.csv`: per-split MSE of the re-evaluated configs
- `xgb_archive.csv`, `rpf_archive.csv`: all evaluated configs on the learner scale in evaluation order, with per-fold MSE
- `search_space.csv`: tuning bounds of the run that produced the archives (log scale where `logscale` is TRUE)

`tuning-report.qmd` analyzes the archives (convergence, CV noise, search space bounds) and the re-evaluation; render with `make report`.

Refit without tuning, using `make_bike_task()` from `R/data_bike.R`. xgboost (`mlr3learners`) needs `encode = TRUE`; rpf (`mlr3extralearners`) takes ~3 min single-threaded, set `nthreads` to speed up:

```r
params <- jsonlite::read_json("results/best_params.json", simplifyVector = TRUE)
learner <- lrn("regr.rpf")
learner$param_set$set_values(.values = params$rpf$param_vals)
learner$train(make_bike_task(preprocess_bike(load_bikeshare())))
```

Or, if you just want the parameters without cloning this repo first:

```r
params <- jsonlite::read_json("https://raw.githubusercontent.com/jemus42/bikeshare-tuning/refs/heads/main/results/best_params.json")
```
