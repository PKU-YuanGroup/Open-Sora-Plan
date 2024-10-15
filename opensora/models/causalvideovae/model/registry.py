class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, model_name):
        def decorator(model_class):
            cls._models[model_name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, model_name):
        return cls._models.get(model_name)