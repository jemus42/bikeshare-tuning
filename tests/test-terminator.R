# Needs a running Redis server; run from the project root with `Rscript tests/test-terminator.R`
library(bbotk)
source("R/terminator.R")
lgr::get_logger("bbotk")$set_threshold("warn")
lgr::get_logger("rush")$set_threshold("warn")

mirai::daemons(3)
rush::rush_plan(n_workers = 3, worker_type = "mirai")

n_finished <- function(fun, n_evals = 40, iters = 5) {
  objective <- ObjectiveRFun$new(
    fun = function(xs) {
      Sys.sleep(0.2)
      list(y = fun())
    },
    domain = ps(x = p_dbl(0, 1)),
    codomain = ps(y = p_dbl(tags = "minimize"))
  )
  instance <- oi_async(objective, terminator = trm_evals_or_stagnation(n_evals, iters = iters))
  opt("async_random_search")$optimize(instance)
  on.exit(instance$rush$reset())
  sum(instance$archive$data$state == "finished")
}

stopifnot(
  "always improving runs to n_evals" = n_finished(\() -as.numeric(Sys.time())) >= 40,
  "constant stops after iters + 1" = n_finished(\() 1) < 15
)
mirai::daemons(0)
message("ok")
