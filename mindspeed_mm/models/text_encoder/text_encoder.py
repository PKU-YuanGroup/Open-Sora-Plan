import importlib


class TextEncoder:
    """
    Instantiate a text encoder model from config.

    Args:
        config (dict): the general config for Text Encoder Model
        {
            (1) args for our feautrues
            "backend": type-str, "hf" or "om",
            "automodel_name": type-str, "AutoModel" or other automodel name,
            
            (2) args for automodel.from_pretrained() of transformers or openmind
            "pretrained_model_name_or_path": type-str, local path or hub path,
            "local_files_only": type-bool,
            ...
        }
    """
    def __init__(self, config):
        self.backend = config.pop("hub_backend")
        self.automodel_name = config.pop("automodel_name")

        # Only huggingface backend is supported, OpenMind backend will be supported soon.
        module = importlib.import_module("transformers")
        automodel = getattr(module, self.automodel_name)
        self.model = automodel.from_pretrained(**config)

    def get_model(self):
        return self.model
