import torch as th

class EasyDict:

    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))

def log_state(state):
    result = []
    
    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    
    return '\n'.join(result)