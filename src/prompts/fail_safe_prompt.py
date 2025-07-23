from typing import Optional, Any

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from loguru import logger
from urllib3.util.retry import Retry

from src.prompts.prompt_templates import PromptTemplates


class PromptNotFound(Exception):
    pass


class FailSafeLangSmithClient:
    """
    Singleton wrapper for LangSmith client that adds fail-safe prompt handling
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            try:
                self._client = Client(retry_config=Retry(
                    total=1,
                    status_forcelist=[],
                    raise_on_status=False)
                )
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith client: {str(e)}")
                raise e

    def pull_prompt(self, prompt_name: str, *args, **kwargs) -> Optional[Any]:
        """
        Get a prompt from LangSmith. If it doesn't exist, create it using the local prompt.
        """
        try:
            res = self._client.pull_prompt(prompt_name)
            raw_template = res.get_prompts()[0][0].prompt.template
            logger.info(f"Pulled {prompt_name} prompt from LangSmith")
            return raw_template.format(**kwargs)
        except Exception as e:
            try:
                template = PromptTemplates().get_prompt_raw(prompt_template_mnemonic=prompt_name, **kwargs)
                self._client.push_prompt(prompt_name,
                                         object=ChatPromptTemplate.from_template(template.template),
                                         is_public=False)
                logger.info(f"Pushing {prompt_name} to LangSmith")
                return template.format(**kwargs)
            except PromptNotFound as e:
                raise e
            except Exception as e:
                logger.error(f"Failed to create prompt {prompt_name}:\n{str(e)}")
                raise e

    def __getattr__(self, name):
        """
        Delegate all other methods to the original client
        """
        return getattr(self._client, name)


def get_prompt_with_fallback(prompt_template_mnemonic: str, **kwargs) -> str:
    """
    Retrieves a prompt using FailSafeLangSmithClient. If that fails,
    logs the error and falls back to the local PromptTemplates.
    Args:
        prompt_template_mnemonic: The identifier for the desired prompt.
        **kwargs: Additional parameters to format the prompt.

    Returns:
        A formatted prompt string.
    """
    try:
        return FailSafeLangSmithClient().pull_prompt(prompt_template_mnemonic, **kwargs)
    except Exception as e:
        logger.error(
            f"Failed to pull prompt from LangSmith for {prompt_template_mnemonic}: {e}. Falling back to local prompt."
        )
        return PromptTemplates().get_prompt(prompt_template_mnemonic, **kwargs)