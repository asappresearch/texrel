import torch


def dump_tensor_dict(msg, tensor_dict):
    """
    dump dict with tensors in. doesnt print the entirety of every tensor...
    """
    print(msg)
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            if (v != 0).long().sum().item() * 3 < v.numel():
                # sparse
                print('nonzerod', k, v.view(-1).nonzero().view(-1)[:20])
            else:
                print(k, v.view(-1)[:20])
        else:
            print(k, v)


def module_is_cuda(module):
    is_cuda_l = []
    for n, params in enumerate(module.parameters()):
        is_cuda_l.append(params.is_cuda)
    is_cuda = is_cuda_l[0]
    for _c in is_cuda_l:
        assert _c == is_cuda
    return is_cuda
