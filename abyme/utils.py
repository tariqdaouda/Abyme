def filter_dict(dict, keys=None):
    """filters a dict to keep on certain keys. if Keys is None, returns the dict"""
    if keys is None :
        return dict
    else :
        return{ key: dict[key] for key in keys }

def prefix_dict_keys(dct, prefix=None):
    """Return a new dict with all keys prefixed"""
    if prefix is None:
        return dct

    return{ prefix + key: value for key, value in dct.items() }

def flatten_dict(dct, prefix="", separator=".", res = None):
    """converts nested dicts into a single one-level dict"""
    if res is None :
        res_dct = {}
    else :
        res_dct = res

    for key, value in dct.items():
        if len(prefix) > 0 :
            kkey = prefix + separator + key
        else :
            kkey = key

        if type(value) is dict :
            res_dct.update(
                flatten_dict(value, prefix=kkey, separator=separator, res = res_dct)
            )
        else :
            res_dct[kkey] = value
    return res_dct

def apply_list_transform(value, transforms):
    """apply a serie of transformations (callables) to a value"""
    if transforms is None :
        return value

    val = value
    for trans in transforms:
        val = trans(val)
    return val
