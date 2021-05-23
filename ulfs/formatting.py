import math


def mean_err_to_str(mean: float, err: float, max_sig: int = 3, err_sds: int = 2,
                    show_err: bool = True, show_mean: bool = True, na_val: str = 'n/a'):
    """
    based on err, will round mean and err to some precision that looks sensible-ish

    eg instead of 12.234+/-0.432 we'd have
    12.2+/-0.5
    """
    if mean != mean:
        return na_val
    prec_err = max_sig
    if err == 0:
        if isinstance(mean, int):
            ret_str = ''
            if show_mean:
                ret_str += f'{mean}'
            if show_err:
                ret_str += f'+/-{err}'
            return ret_str
        else:
            ret_str = ''
            if show_mean:
                ret_str += f'%.{prec_err}f' % mean
            if show_err:
                ret_str += f'+/-%.{prec_err}f' % err
            return ret_str
    prec_err = min(max_sig, - math.floor(math.log10(err)) + err_sds - 1)
    if prec_err >= 0:
        mean_formatted = f'%.{prec_err}f' % mean
        err_formatted = f'%.{prec_err}f' % err
    else:
        multiplier = round(math.pow(10, - prec_err))
        mean_formatted = str(int(mean / multiplier) * multiplier)
        err_formatted = str(int(err / multiplier) * multiplier)
    ret_str = ''
    if show_mean:
        ret_str += mean_formatted
    if show_err:
        ret_str += f'+/-{err_formatted}'
    return ret_str
