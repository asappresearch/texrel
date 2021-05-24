from torch import nn, optim

from ulfs import reptile_lib


def test_reptile_2():
    meta_lr = 0.1
    inner_steps = 5

    model = nn.Linear(1, 1)
    opt = optim.Adam(model.parameters())
    crit = nn.MSELoss()
    trainer = reptile_lib.ReptileTrainer(
        model=model, inner_opt=opt, meta_lr=meta_lr, inner_crit=crit, inner_steps=inner_steps)
    model_params = {k: p.clone() for k, p in model.named_parameters()}

    [p.data.zero_() for k, p in model.named_parameters()]
    model_params = {k: p.clone() for k, p in model.named_parameters()}
    assert sum([p.abs().sum().item() for k, p in model.named_parameters()]) == 0

    trainer.reset_inner()
    model_params = {k: p.clone() for k, p in model.named_parameters()}
    for k, p in model.named_parameters():
        assert (p.data == model_params[k].data).all()

    for k, p in trainer.outer_grad.items():
        assert (p == 0).all()

    for k, p in model.named_parameters():
        p.data.fill_(1.23)
    model_params = {k: p.clone() for k, p in model.named_parameters()}

    for k, p in model.named_parameters():
        assert (p.data == 1.23).all()

    trainer.outer_backward()
    for k, p in trainer.outer_grad.items():
        assert (p != 0).all()
        assert p == dict(model.named_parameters())[k] - trainer.outer_params[k]
    assert trainer.outer_grad_step_count == 1

    outer_params_before = {k: p.clone() for k, p in trainer.outer_params.items()}
    trainer.outer_opt_step()
    for k, p in trainer.outer_params.items():
        assert (p.data == outer_params_before[k] + meta_lr * trainer.outer_grad[k]).all()
        assert p.data != outer_params_before[k]

    for k, p in trainer.outer_grad.items():
        assert (p != 0).all()
    trainer.outer_zero_grad()

    for k, p in trainer.outer_grad.items():
        assert (p == 0).all()
