import importlib


def base_build_fn(module, cls, params):
    return getattr(importlib.import_module(
                    module, package=None), cls)(**params)


def build_from_cfg(config):
    module, cls = config['name'].rsplit(".", 1)
    params = config.get('params', {})
    return base_build_fn(module, cls, params)
