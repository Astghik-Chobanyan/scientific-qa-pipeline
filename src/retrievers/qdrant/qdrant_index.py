import os
from typing import List

from loguru import logger
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

from src.config.config_manager import ConfigManager
from src.config.model_loader import model_loader

load_dotenv()



class QdrantIndex:
    def __init__(self, embedding_model_name: str, cohere_top_n: int = 4):
        """
        Initializes the QdrantIndex with a Qdrant client and a single embedding model.
        Both the default and summary embeddings will use the same model.

        Args:
            embedding_model_name (str): Name of the SentenceTransformer model for embeddings.
        """
        self.client = QdrantClient(
            api_key=os.getenv("QDRANT_API_KEY"),
            url=os.getenv("QDRANT_URL")
        )
        self.embedding_model = model_loader.load_model(embedding_model_name, SentenceTransformer)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        self.summary_embedding_model = self.embedding_model
        self.cohere_top_n = cohere_top_n
        self.summary_vector_size = self.vector_size

    def create_collection_if_not_exists(self, collection_name: str):
        """
        Creates a Qdrant collection if it doesn't already exist.
        The collection is configured to store two named vectors:
            - "default" for full text embeddings.
            - "summary" for summary embeddings.
        """
        if not self.client.collection_exists(collection_name):
            vectors_config = {
                "default": VectorParams(size=self.vector_size, distance=Distance.COSINE)
            }
            vectors_config["summary"] = VectorParams(size=self.summary_vector_size, distance=Distance.COSINE)

            self.client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
            logger.info(f"Collection '{collection_name}' created with vectors: {list(vectors_config.keys())}.")
        else:
            logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")

    def upload_points(self, collection_name: str, points: List[dict], batch_size: int = 100):
        """
        Uploads points (embeddings with payload) to a Qdrant collection in batches.

        Args:
            collection_name (str): Name of the Qdrant collection.
            points (List[dict]): List of point dictionaries to upload.
            batch_size (int): Maximum number of points to send in a single upsert call.
        """
        self.create_collection_if_not_exists(collection_name)

        try:
            count_response = self.client.count(collection_name=collection_name)
            existing_count = count_response.count
        except Exception as e:
            logger.error(f"Error retrieving count for collection '{collection_name}': {e}")
            existing_count = 0

        for idx, point in enumerate(points):
            point['id'] = existing_count + idx + 1

        if points:
            total = len(points)
            for i in range(0, total, batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(collection_name=collection_name, points=batch)
                logger.info(
                    f"Uploaded batch {existing_count + i + 1} to {existing_count + i + len(batch)} of {total} points to collection '{collection_name}'.")
            logger.info(
                f"Uploaded total {total} points to collection '{collection_name}' (starting from {existing_count + 1}).")
        else:
            logger.info("No points to upload.")

    def query_collection(self, collection_name: str, query: str, top_k: int = 3, search_type: str = "default"):
        """
        Queries a Qdrant collection using the specified vector field.
        The query is encoded using the appropriate embedding model based on `search_type`.

        Args:
            collection_name (str): The target Qdrant collection.
            query (str): The query string.
            top_k (int): Number of top results to return.
            search_type (str): Either "default" for full text or "summary" for summary embeddings.

        Returns:
            List[dict]: A list of query result objects, reranked with the cohere client.
        """
        # Choose the appropriate vector field (both use the same model).
        if search_type == "default":
            vector_field = "default"
            query_vector = self.embedding_model.encode(query).tolist()
        elif search_type == "summary":
            vector_field = "summary"
            query_vector = self.summary_embedding_model.encode(query).tolist()
        else:
            logger.error(f"Unknown search_type '{search_type}'. Falling back to default.")
            vector_field = "default"
            query_vector = self.embedding_model.encode(query).tolist()

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            using=vector_field
        )

        document_list = [{"content": point.payload.get('page_content', ''),
                          "metadata": point.payload.get('metadata', {}),
                          "score": point.score}
                         for point in results.points]

        return document_list

if __name__ == '__main__':
    # Example usage
    index = QdrantIndex(embedding_model_name=ConfigManager().embedding_model_name)
    topic_questions_examples = {
        "Diffusion_Models_in_Computer_Vision": [
            "What are refinement and quotient types in DHOL, and how does the system avoid changes in representation when introducing them?",
            "Why do the authors argue that answer matching is a better evaluation method than multiple-choice questions for language models?",
            "Describe how AnyI2V enables motion-controlled video generation using a training-free approach."
        ],
        "Graph_Neural_Networks_(GNNs)_for_Molecular_Property_Prediction": [],
        "Large_Language_Models_for_Mathematical_Reasoning": [],
        "Transformer_Models_for_Protein_Folding": []

    }
    for topic, questions in topic_questions_examples.items():
        print(f"Topic: {topic}")
        for question in questions:
            print(f"  Question: {question}")
            results = index.query_collection(collection_name=topic, query=question, top_k=3)
            print(f"  Results: {results}\n")
        print("\n")