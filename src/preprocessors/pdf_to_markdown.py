import json
import os

from docling.document_converter import DocumentConverter


def convert_all_pdfs_to_markdown(root_path: str):
    converter = DocumentConverter()

    for sub_folder in os.listdir(root_path):
        if sub_folder == ".DS_Store":
            continue

        sub_folder_path = os.path.join(root_path, sub_folder)
        pdfs_metadata_path = os.path.join(sub_folder_path, 'pdfs_metadata.json')
        save_root_path = os.path.join(sub_folder_path, 'markdowns')

        if os.path.exists(save_root_path):
            print(f"Markdowns already exist in {save_root_path}. Skipping conversion.")
            continue

        os.makedirs(save_root_path, exist_ok=True)

        with open(pdfs_metadata_path, 'r') as f:
            pdfs_metadata = json.load(f)

        for pdf_metadata in pdfs_metadata:
            print(f"Title: {pdf_metadata['title']}")
            print(f"PDF URL: {pdf_metadata['pdf_url']}")
            print(f"Filepath: {pdf_metadata['filepath']}\n")

            result = converter.convert(pdf_metadata['pdf_url'])
            markdown = result.document.export_to_markdown()

            filename = os.path.basename(pdf_metadata['filepath']).replace('.pdf', '.md')
            markdown_file_path = os.path.join(save_root_path, filename)

            with open(markdown_file_path, 'w') as md_file:
                md_file.write(markdown)


if __name__ == '__main__':
    convert_all_pdfs_to_markdown("../data")
