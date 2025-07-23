from pathlib import Path

from llama_index.core.prompts import PromptTemplate

from src.utils.singleton_meta import SingletonMeta


class PromptTemplates(metaclass=SingletonMeta):

    def __init__(self):
        self.prompt_templates_path = Path(__file__).resolve().parent / 'templates'
        if not self.prompt_templates_path.exists():
            raise ValueError(f"Prompt path not found: {self.prompt_templates_path}.")
        self.prompt_templates = {}

        for prompt_template_file in self.prompt_templates_path.rglob('*.prompt'):
            self.prompt_templates[prompt_template_file.stem] = self._load_prompt_template_text(prompt_template_file)

    def _load_prompt_template_text(self, prompt_file: Path):
        with open(prompt_file, 'r') as f:
            return PromptTemplate(f.read())

    def get_prompt_template(self, prompt_template_mnemonic: str):
        if prompt_template_mnemonic not in self.prompt_templates:
            raise ValueError(f"Prompt template mnemonic '{prompt_template_mnemonic}' not found.")
        return self.prompt_templates[prompt_template_mnemonic]

    def get_prompt(self, prompt_template_mnemonic: str, **kwargs):
        return self.get_prompt_template(prompt_template_mnemonic).format(**kwargs)

    def get_prompt_raw(self, prompt_template_mnemonic: str, **kwargs):
        return self.get_prompt_template(prompt_template_mnemonic)