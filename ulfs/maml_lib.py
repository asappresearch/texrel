from typing import Tuple
import copy
import torch


class MAMLfy(object):
    def __init__(self, inner_model, inner_crit, inner_lr):
        self.model = inner_model
        self.model_dash = copy.deepcopy(inner_model)
        self.inner_crit = inner_crit
        self.inner_lr = inner_lr

    def parameters(self):
        return self.model.parameters()

    def cuda(self):
        self.model = self.model.cuda()
        self.model_dash = self.model_dash.cuda()
        return self

    def reset_model_dash(self):
        """
        copy model parameters to model_dash parameters, and to theta_dash
        """
        self.theta_dash = []
        for i, param in enumerate(self.model.parameters()):
            self.theta_dash.append(param)

        for i, param in enumerate(self.model_dash.parameters()):
            param.data = self.theta_dash[i]

    def inner_train(self, inner_train_X: torch.Tensor, inner_train_Y: torch.Tensor) -> Tuple[
            float, torch.Tensor]:
        inner_train_out = self.model_dash(inner_train_X)
        inner_loss = self.inner_crit(inner_train_out, inner_train_Y)
        grads = torch.autograd.grad(inner_loss, self.model_dash.parameters(), create_graph=True)

        for i, param in enumerate(self.model.parameters()):
            self.theta_dash[i] = self.theta_dash[i] - self.inner_lr * grads[i]

        for i, param in enumerate(self.model_dash.parameters()):
            param.data = self.theta_dash[i]

        return inner_loss.item(), inner_train_out

    def meta_train(self, inner_test_X: torch.Tensor, inner_test_Y: torch.Tensor) -> Tuple[
            float, torch.Tensor]:
        inner_test_out = self.model_dash(inner_test_X)
        loss_meta = self.inner_crit(inner_test_out, inner_test_Y)

        grads_meta_dash = torch.autograd.grad(loss_meta, self.model_dash.parameters())
        grads_meta = torch.autograd.grad(self.theta_dash, self.model.parameters(), grads_meta_dash)

        for i, p in enumerate(self.model.parameters()):
            p.grad = grads_meta[i].clone() if p.grad is None else p.grad + grads_meta[i].clone()

        return loss_meta.item(), inner_test_out
