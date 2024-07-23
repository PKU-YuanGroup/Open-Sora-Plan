import importlib

Module = str
MODULES_BASE = "opensora.models.causalvideovae.model.modules."

def resolve_str_to_obj(str_val, append=True):
    if append:
        str_val = MODULES_BASE + str_val
    module_name, class_name = str_val.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def create_instance(module_class_str: str, **kwargs):
    module_name, class_name = module_class_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**kwargs)