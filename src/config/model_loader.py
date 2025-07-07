from typing import Callable, Any

class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.models = {}
        return cls._instance

    def load_model(self, model_name: str, Loader: Callable[[str], Any]):
        if model_name not in self.models:
            self.models[model_name] = Loader(model_name)
            print(f"Model loaded: {model_name}")
        return self.models[model_name]

model_loader = ModelLoader()