"""
check weights and grads

(checks weights not zero, and grads not zero
"""
import math


class WeightsGradsChecker(object):
    def __init__(self, dumper):
        self.dumper = dumper

    def check(self, params, objects_to_dump):
        res = ''
        params = list(params)
        res += self.check_grads(params, objects_to_dump)
        res += ' ' + self.check_weights(params, objects_to_dump)
        return res

    def check_weights(self, params, objects_to_dump):
        abs_max = 0
        non_zero_weights = 0
        weights_sum = 0
        # weights_count = 0
        for _p in params:
            abs_max = max(abs_max, _p.abs().max().item())
            _non_zero_weights = _p.nonzero().size()[0]
            non_zero_weights += _non_zero_weights
            weights_sum += _p.abs().sum().item()
        if non_zero_weights == 0:
            print('non_zero_weights', non_zero_weights)
            print('abs_max', abs_max)
            print('weights_sum', weights_sum)
            print('len(params)', len(list(params)))
        weights_avg = weights_sum / non_zero_weights
        res_string = f'weight_abs_max={abs_max:.3f} nonzeros={non_zero_weights} absavg={weights_avg:.3f}'
        if abs_max < 1e-8 or non_zero_weights == 0 or weights_avg != weights_avg or math.isnan(weights_avg):
            print('weights look strange', res_string)
            print('weights_sum', weights_sum)
            print('non_zero_weights', non_zero_weights)
            if self.dumper is not None:
                self.dumper.dump(objects_to_dump)
            raise Exception('weights abs max < 1e-8')
        return res_string

    def check_grads(self, params, objects_to_dump):
        grad_abs_max = 0
        grad_abs_sum = 0
        total_elem = 0
        for _p in params:
            grad_abs_max = max(grad_abs_max, _p.grad.abs().max().item())
            grad_abs_sum += _p.grad.abs().sum().item()
            total_elem += _p.grad.numel()
        grad_abs_avg = grad_abs_sum / total_elem
        res_string = f'grads_abs_max={grad_abs_max} absavg={grad_abs_avg}'
        if grad_abs_max < 1e-8 or grad_abs_avg != grad_abs_avg or math.isnan(grad_abs_avg):
            print('grads look strange', res_string)
            if self.dumper is not None:
                self.dumper.dump(objects_to_dump)
            raise Exception('grad abs max < 1e-8')
        return res_string
