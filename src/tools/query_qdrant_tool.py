from typing import Annotated, List

from langchain_core.tools import tool
from loguru import logger

from src.config.config_manager import ConfigManager
from src.retrievers.qdrant.qdrant_index import QdrantIndex


@tool
def query_qdrant_tool(query: Annotated[str, "The search query"],
                      topic_name: Annotated[str, "The name of the topic whose vector database will be queried"]
                      ) -> List:
    """
    Tool specifically designed to execute a similarity search against a Qdrant vector index to retrieve relevant documents.
    """
    try:
        logger.info(f'Qdrant Query | Topic: {topic_name}')
        page_contents = ''
        top_k = ConfigManager().qdrant_top_k
        index = QdrantIndex(ConfigManager().embedding_model_name)
        results = index.query_collection(collection_name=topic_name, query=query, top_k=top_k)
        # if results.points:
        #     page_contents = '\n'.join([result.payload['page_content'] for result in results.points])
        # logger.info(f'Qdrant Response | Topic: {topic_name} | Response: {page_contents}')
        return results
    except Exception as e:
        logger.error(f"Qdrant Query Failed | Topic: {topic_name} | Error: {str(e)}")
        return ""