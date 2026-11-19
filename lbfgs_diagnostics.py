"""Record why the pinned PyTorch L-BFGS implementation stopped."""

import torch


class MonitoredLBFGS(torch.optim.LBFGS):
    """Track applied updates separately from temporary line-search trials."""

    def __init__(self, params, **options):
        super().__init__(params, **options)
        self.in_trial = False
        self.updates = 0

    def _directional_evaluate(self, closure, x, t, d):
        self.in_trial = True
        try:
            return super()._directional_evaluate(closure, x, t, d)
        finally:
            self.in_trial = False

    def _add_grad(self, step_size, update):
        super()._add_grad(step_size, update)
        if not self.in_trial:
            self.updates += 1

    def diagnostics(self, final_loss, final_grad):
        # Match the stopping-check order in the pinned PyTorch optimizer.
        first = self.param_groups[0]
        state = self.state[first["params"][0]]
        iterations = state["n_iter"]
        evaluations = state["func_evals"]
        grad_max = final_grad.abs().max().item()
        change = first["tolerance_change"]
        if iterations == 0:
            reason = "initial_gradient_tolerance"
        elif self.updates < iterations:
            reason = "directional_derivative_tolerance"
        elif iterations == first["max_iter"]:
            reason = "iteration_limit"
        elif evaluations >= first["max_eval"]:
            reason = "evaluation_limit"
        elif grad_max <= first["tolerance_grad"]:
            reason = "gradient_tolerance"
        elif (state["d"] * state["t"]).abs().max().item() <= change:
            reason = "step_size_tolerance"
        elif abs(final_loss - state["prev_loss"]) < change:
            reason = "loss_change_tolerance"
        else:
            reason = "undetermined"
        return {
            "lbfgs_step_calls": 1,
            "lbfgs_iterations": iterations,
            "lbfgs_applied_updates": self.updates,
            "lbfgs_closure_evaluations": evaluations,
            "lbfgs_stop_reason": reason,
        }
