import json
import os.path
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from typing import List

from src.config.config_manager import ConfigManager
from src.retrievers.qdrant.chunker import TextChunker
from src.retrievers.qdrant.qdrant_index import QdrantIndex


load_dotenv()

def get_doc_metadata(doc_metadata, markdown_file: str) -> dict:
    """
    Retrieves metadata for a given markdown file from the provided doc_metadata dictionary.
    If the markdown file is not found in the metadata, returns an empty dictionary.

    Args:
        doc_metadata (dict): Dictionary containing metadata for documents.
        markdown_file (str): The name of the markdown file to retrieve metadata for.

    Returns:
        dict: Metadata for the markdown file or an empty dictionary if not found.
    """
    doc_name = markdown_file.name.split(".md")[0]
    for info in doc_metadata:
        if info.get("pdf_url", "").endswith(doc_name):
            return info

def process_site_collection(collection_name: str, website_dirs: List[Path], uploader: QdrantIndex) -> None:
    """
    Processes markdown files for a given site configuration by reading all markdown files from
    the provided website directories and uploading the corresponding points to Qdrant.
    """
    logger.info(f"Processing collection '{collection_name}' from directories: {website_dirs}")

    all_points = []
    base_id = 0
    with open(os.path.join(root_dir, sub_dirs, "pdfs_metadata.json"), 'r') as f:
        pdfs_metadata = json.load(f)
    # Process each directory found for this collection.
    for website_dir in website_dirs:
        for file in website_dir.rglob("*.md"):
            if file.is_file():
                points = process_markdown_file(file, uploader, base_id, pdfs_metadata)
                base_id += 1
                all_points.extend(points)

    if all_points:
        uploader.upload_points(collection_name, all_points)
        logger.info(f"Uploaded {len(all_points)} points for collection '{collection_name}'")
    else:
        logger.info(f"No markdown files processed for collection '{collection_name}'.")


def process_markdown_file(file_path: Path, uploader: QdrantIndex, start_id: int, doc_metadata) -> List[dict]:
    """
    Reads a markdown file, chunks it using TextChunker, and creates points for Qdrant.
    Each point will contain:
      - "page_content": the chunked text.
      - "metadata": file name, URL (from metadata.json or derived), and PDF name (if any), summary text (from metadata.json if available
    Also computes two embeddings:
      - A default vector for the full text.
      - A summary vector for the summary text.

    Returns a tuple of the list of points and the updated unique ID counter.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return [], start_id

    metadata = get_doc_metadata(doc_metadata, file_path)

    chunks = TextChunker.header_aware_chunking(text=content, metadata=str(file_path))
    summary_vector = uploader.summary_embedding_model.encode(metadata.get("summary", "") if metadata else "")

    points = []
    for chunk in chunks:
        default_vector = uploader.embedding_model.encode(chunk)

        point = {
            "id": start_id,
            "vector": {
                "default": default_vector.tolist(),
                "summary": summary_vector.tolist()
            },
            "payload": {
                "page_content": chunk,
                "metadata": {
                    "article_title": metadata.get("title", "") if metadata else "",
                    "url": metadata.get("pdf_url", "") if metadata else "",
                    "pdf_name": metadata.get("pdf_url", "").split("/")[-1] if metadata else "",
                    "summary": metadata.get("summary", "") if metadata else "",
                    "authors": metadata.get("authors", []) if metadata else [],
                    "published": metadata.get("published", "") if metadata else "",
                },
            }
        }

        points.append(point)
        start_id += 1

    return points


if __name__ == '__main__':
    root_dir = Path("/Users/astghikchobanyan/Desktop/scientific-qa-pipeline/src/data")
    uploader = QdrantIndex(embedding_model_name=ConfigManager().embedding_model_name)
    for sub_dirs in root_dir.iterdir():
        if sub_dirs.is_dir() and sub_dirs.name != ".DS_Store":
            collection_name = sub_dirs.name.split("arxiv_pdfs_")[-1]
            markdown_dirs = list(sub_dirs.glob("**/markdowns"))
            if not markdown_dirs:
                logger.warning(f"No markdown directories found for collection '{collection_name}'. Skipping.")
                continue
            with open(os.path.join(root_dir, sub_dirs, "pdfs_metadata.json"), 'r') as f:
                pdfs_metadata = json.load(f)
            logger.info(f"Processing collection '{collection_name}' with directories: {markdown_dirs}")
            process_site_collection(collection_name, markdown_dirs, uploader)
            # for markdown_file in os.listdir(markdown_dirs[0]):
            #     process_markdown_file(file_path=markdown_dirs[0].joinpath(markdown_file), uploader=uploader, start_id=1, doc_metadata=pdfs_metadata)
