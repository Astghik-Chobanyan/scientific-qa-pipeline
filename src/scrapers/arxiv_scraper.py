import json

import arxiv
import os
import requests


def download_arxiv_pdfs(topic: str, max_results: int = 10, output_dir: str = "./arxiv_pdfs"):
    """
    Download the latest PDFs from arXiv based on a topic query.

    Parameters:
        topic (str): Search query topic for arXiv.
        max_results (int): Maximum number of results to download.
        output_dir (str): Directory to save downloaded PDFs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    print(f"Searching arXiv for: {topic} (max results: {max_results})")

    pdf_urls = []
    for result in search.results():
        try:
            pdf_url = result.pdf_url
            filename = f"{result.get_short_id().replace('/', '_')}.pdf"
            filepath = os.path.join(output_dir, filename)
            pdf_urls.append({
                "title": result.title,
                "pdf_url": pdf_url,
                "filepath": filepath,
                "summary": result.summary,
                "authors": [author.name for author in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
            })
            print(f"Downloading: {result.title}\nFrom: {pdf_url}")
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Saved to: {filepath}\n")

        except Exception as e:
            print(f"Failed to download {result.title}: {e}\n")
    with open(os.path.join(output_dir, "pdfs_metadata.json"), "w") as f:
        json.dump(pdf_urls, f, indent=4)

# Example usage
if __name__ == "__main__":
    list_of_topics = [
        "Diffusion Models in Computer Vision",
        "Graph Neural Networks (GNNs) for Molecular Property Prediction",
        "Transformer Models for Protein Folding",
        "Large Language Models for Mathematical Reasoning"
    ]
    for topic in list_of_topics:
        download_arxiv_pdfs(topic=topic, max_results=10, output_dir=f"../data/arxiv_pdfs_{topic.replace(' ', '_')}")
