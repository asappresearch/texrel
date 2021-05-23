from typing import Tuple, List, Iterable
import torch
from torch import nn, optim


class ReptileTrainer(object):
    def __init__(
            self, model: nn.Module, inner_opt: optim.Optimizer, meta_lr: float,
            inner_crit, inner_steps: int):
        self.model = model
        self.inner_opt = inner_opt
        self.meta_lr = meta_lr
        self.inner_crit = inner_crit
        self.inner_steps = inner_steps
        self.outer_grad_step_count = 0

        self.outer_params = {k: p.clone().detach() for k, p in model.named_parameters()}
        self.inner_params = dict(self.model.named_parameters())
        self.outer_grad = {k: p.clone().detach().zero_() for k, p in model.named_parameters()}

    def cuda(self):
        self.outer_params = {k: p.cuda() for k, p in self.outer_params.items()}
        self.outer_grad = {k: p.cuda() for k, p in self.outer_grad.items()}

    def reset_inner(self):
        for k, p in self.outer_params.items():
            self.inner_params[k].data[:] = p.clone().detach()

    def train_inner_batch(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        inner_losses_l = []
        for it in range(self.inner_steps):
            out_logits = self.model(X)
            inner_train_loss = self.inner_crit(out_logits, Y)
            inner_losses_l.append(inner_train_loss.item())
            self.inner_opt.zero_grad()
            inner_train_loss.backward()
            self.inner_opt.step()
        return out_logits.detach(), inner_losses_l

    def inner_pred(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred_logits = self.model(X)
        return pred_logits

    def outer_zero_grad(self) -> None:
        for k, p in self.outer_grad.items():
            p.zero_()
        self.outer_grad_step_count = 0

    def outer_backward(self) -> None:
        for k, p in self.outer_grad.items():
            self.outer_grad[k] = p + (self.inner_params[k].detach() - self.outer_params[k])
        self.outer_grad_step_count += 1

    def outer_opt_step(self) -> None:
        for k, p in self.outer_params.items():
            self.outer_params[k] = p + self.meta_lr * self.outer_grad[k] / self.outer_grad_step_count

    def train_outer_batch(
            self, inner_batches: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[
                List[torch.Tensor], List[torch.Tensor], float, float, float]:
        self.outer_zero_grad()
        train_losses_l = []
        test_losses_l = []
        inner_loss_change_l = []
        train_inner_preds_l = []
        test_inner_preds_l = []
        for train_X, train_Y, test_X, test_Y in inner_batches:
            self.reset_inner()
            train_pred, inner_losses_l = self.train_inner_batch(X=train_X, Y=train_Y)
            inner_loss_change_l.append(inner_losses_l[0] - inner_losses_l[-1])
            train_inner_preds_l.append(train_pred)
            train_losses_l.append(inner_losses_l[-1])
            self.outer_backward()
            test_out_logits = self.inner_pred(X=test_X)
            test_inner_preds_l.append(test_out_logits)
            test_loss = self.inner_crit(test_out_logits, test_Y)
            test_losses_l.append(test_loss.item())
        self.outer_opt_step()
        inner_train_loss = sum(train_losses_l) / len(train_losses_l)
        inner_test_loss = sum(test_losses_l) / len(test_losses_l)
        inner_loss_change = sum(inner_loss_change_l) / len(inner_loss_change_l)
        return (
            train_inner_preds_l, test_inner_preds_l,
            inner_loss_change, inner_train_loss, inner_test_loss)
