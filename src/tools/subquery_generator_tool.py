import json
import traceback
from typing import Annotated

from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from loguru import logger

from src.config.config_manager import ConfigManager
from src.prompts.fail_safe_prompt import get_prompt_with_fallback


@tool
def subquery_generator_tool(query: Annotated[str, 'The user query']) -> list[str] | str:
    """
    Generates subqueries based on the user query.
    """
    try:
        logger.info(f'[START] Subquery Generator')
        prompt = get_prompt_with_fallback(
            prompt_template_mnemonic="subquery_generator",
            user_query=query,
        )
        llm = ChatOllama(model=ConfigManager().ollama_thinking_model_name, format="json")
        response = llm.invoke([
                (
                    "system",
                     "You are an AI assistant that analyzes the user query and generates subqueries from it if possible. "
                ),
                ("user", prompt)
            ]
        )
        subqueries = json.loads(response.content)['subqueries']
        logger.info(f'[COMPLETE] Subquery Generator | Response: {subqueries}')
        return subqueries
    except Exception as e:
        logger.error(f"[FAIL] Subquery Generator | Error: {str(e)}")
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        return ''