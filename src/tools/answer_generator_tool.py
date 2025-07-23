import json
import traceback
from typing import Annotated, List

from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from loguru import logger

from src.config.config_manager import ConfigManager
from src.prompts.fail_safe_prompt import get_prompt_with_fallback


@tool
def answer_generator_tool(query: Annotated[str, 'The user query'], retrieved_data: List) -> list[str] | str:
    """
    Answer the user query based on the retrieved data.
    """
    try:
        logger.info(f'[START] Answer Generator')
        prompt = get_prompt_with_fallback(
            prompt_template_mnemonic="answer_generator",
            query=query,
            results=retrieved_data,
        )
        llm = ChatOllama(model=ConfigManager().ollama_thinking_model_name, format="json")
        response = llm.invoke([
                (
                    "system",
                     "You are an AI assistant that answers user query related to the scientific research topics."
                     "Your output must be in JSON format with a single key 'answer'."
                ),
                ("user", prompt)
            ]
        )
        topic_names = json.loads(response.content)['answer']
        logger.info(f'[COMPLETE] Answer Generator')
        return topic_names
    except Exception as e:
        logger.error(f"[FAIL] Topic Extractor | Error: {str(e)}")
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        return ''