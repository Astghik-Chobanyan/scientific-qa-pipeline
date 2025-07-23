import json
import traceback
from typing import Annotated

from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from loguru import logger

from src.config.config_manager import ConfigManager
from src.prompts.fail_safe_prompt import get_prompt_with_fallback

TOPIC_NAMES = [
    "Diffusion Models in Computer Vision",
    "Graph Neural Networks (GNNs) for Molecular Property Prediction",
    "Transformer Models for Protein Folding",
    "Large Language Models for Mathematical Reasoning"
]


@tool
def topic_extractor_tool(query: Annotated[str, 'The user query']) -> list[str] | str:
    """
    Understand the topic of research based on the user query and return it.
    """
    try:
        logger.info(f'[START] Topic Extractor')
        prompt = get_prompt_with_fallback(
            prompt_template_mnemonic="topic_extractor",
            user_query=query,
            predefined_topics=TOPIC_NAMES
        )
        llm = ChatOllama(model=ConfigManager().ollama_model_name, format="json")
        response = llm.invoke([
                (
                    "system",
                     "You are an AI assistant that analyzes the user query and understands the topic of the research."
                     "Your output must be in JSON format with a single key 'topic_to_refer' containing a string of the topic name."
                ),
                ("user", prompt)
            ]
        )
        topic_names = json.loads(response.content)['topic_name']
        logger.info(f'[COMPLETE] Topic Extractor | Response: {topic_names}')
        return topic_names
    except Exception as e:
        logger.error(f"[FAIL] Topic Extractor | Error: {str(e)}")
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        return ''