import ast
import os
from typing import Annotated

from langchain_core.tools import tool
from loguru import logger
from tavily import TavilyClient

from src.config.config_manager import ConfigManager


@tool
def tavily_search_tool(query: Annotated[str, "The search query"]) -> str:
    """
    Tool designed for domain-specific search using Tavily.
    """
    try:
        logger.info(f'[START] Tavily Search for query: {query}')
        tavily_client = TavilyClient(api_key=os.environ.get('TAVILY_API_KEY'))
        search_results = tavily_client.get_search_context(query,
                                                          search_depth="advanced",
                                                          include_domains=["https://arxiv.org/"],
                                                          max_results=ConfigManager().tavily_search_max_results)
        parsed_results = ast.literal_eval(search_results)
        aggregated_search_results = '\n'.join(f'Source: {result_group["url"]}\nContent: {result_group["content"]}' for result_group in parsed_results)
        logger.info(f'[COMPLETE] Tavily Search for Query: {query} | Response: {aggregated_search_results}')
        return aggregated_search_results
    except Exception as e:
        logger.error(f"[FAIL] Tavily Search for Query: {query} | Error: {str(e)}")
        return ''
