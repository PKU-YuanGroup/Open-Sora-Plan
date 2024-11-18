import importlib


class Tokenizer:
    """
    Instantiate a tokenizer from config.

    Args:
        config (dict): the general config for Tokenizer
        {
            (1) args for our feautrues
            "backend": type-str, "hf" or "om",
            "autotokenizer_name": type-str, "AutoTokenizer" or other autotokenizer name,

            (2) args for autotokenizer.from_pretrained() of transformers or openmind
            "pretrained_model_name_or_path": type-str, local path or hub path,
            "local_files_only": type-bool,
            ...
        }
    """

    def __init__(self, config):
        if not isinstance(config, dict):
            config = config.to_dict()
        self.backend = config.pop("hub_backend")
        self.autotokenizer_name = config.pop("autotokenizer_name")
        config["pretrained_model_name_or_path"] = config.pop("from_pretrained")

        # Only huggingface backend is supported, OpenMind backend will be supported soon.
        module = importlib.import_module("transformers")
        autotokenizer = getattr(module, self.autotokenizer_name)
        self.tokenizer = autotokenizer.from_pretrained(**config)

    def get_tokenizer(self):
        return self.tokenizer
