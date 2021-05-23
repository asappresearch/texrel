def assert_values_empty_none_zero(dict):
    for k, v in dict.items():
        if v is None or v == '' or v == 0:
            continue
        raise Exception(f'error, key {k} passed to function has non-empty value {v}')
