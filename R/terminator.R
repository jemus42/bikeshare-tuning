# bbotk's TerminatorStagnation reads archive$data, which for async archives also holds
# queued/running rows (NA y) grouped by state; workers then die on `while (!is_terminated)`
TerminatorStagnationAsync <- R6::R6Class(
  "TerminatorStagnationAsync",
  inherit = bbotk::TerminatorStagnation,
  public = list(
    is_terminated = function(archive) {
      pv <- self$param_set$values
      finished <- archive$data_with_state(states = "finished")
      if (nrow(finished) <= pv$iters) {
        return(FALSE)
      }
      # Runs on rush workers, which do not attach data.table
      data.table::setorderv(finished, "timestamp_ys")
      # Flip sign so smaller is better regardless of direction
      y <- finished[[archive$cols_y]] * archive$codomain$direction
      n <- length(y)
      min(y[(n - pv$iters + 1):n]) >= min(y[seq_len(n - pv$iters)]) - pv$threshold
    }
  )
)

# Stop at n_evals, or once `iters` finished evals bring no improvement
trm_evals_or_stagnation <- function(n_evals, iters = 200, threshold = 0) {
  stagnation <- TerminatorStagnationAsync$new()
  stagnation$param_set$set_values(iters = iters, threshold = threshold)
  trm("combo", list(trm("evals", n_evals = n_evals, k = 0), stagnation), any = TRUE)
}
