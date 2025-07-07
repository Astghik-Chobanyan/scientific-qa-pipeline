from pathlib import Path
from src.utils.singleton_meta import SingletonMeta


class ConfigManager(metaclass=SingletonMeta):

    def __init__(self):
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.qdrant_top_k = 5

    def configure(self, config_dict):
        for key, value in config_dict.items():
            self.__dict__[key] = value

    @staticmethod
    def get_project_root():
        return Path(__file__).parent.parent.parent.resolve()