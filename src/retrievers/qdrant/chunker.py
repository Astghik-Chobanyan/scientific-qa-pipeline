import re
from typing import List
from loguru import logger


class TextChunker:
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens by splitting on whitespace."""
        return len(text.split())

    @staticmethod
    def chunk_text_with_overlap(tokens: List[str], max_tokens: int = 1024, overlap: int = 256) -> List[str]:
        """Split a list of tokens into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk = tokens[start:end]
            chunks.append(" ".join(chunk))
            if end == len(tokens):
                break
            start += max_tokens - overlap  # slide the window with overlap
        return chunks

    @staticmethod
    def header_aware_chunking(text: str, max_tokens: int = 1024, merge_margin: int = 50, metadata: str = '') -> List[
        str]:
        """
        Splits markdown text into chunks in a header-aware manner, but only if the text exceeds 5000 tokens.
        For texts with ≤5000 tokens, the whole text is returned as a single chunk.
        For very large texts (>5000 tokens), the chunk parameters are adjusted.
        """
        total_tokens = TextChunker.count_tokens(text)
        if total_tokens <= max_tokens:
            # No need to split if the document is not large.
            return [text]

        logger.info(
            f"Large document detected: {total_tokens} tokens for file {metadata}. Using large chunk parameters.")
        # Split text at markdown headers (lines beginning with 1-3 '#' characters).
        header_regex = re.compile(r'^(#{1,3}\s+)', re.MULTILINE)
        sections = []
        last_idx = 0

        for match in header_regex.finditer(text):
            start = match.start()
            if start != last_idx:
                section = text[last_idx:start].strip()
                if section:
                    sections.append(section)
            last_idx = start

        final_section = text[last_idx:].strip()
        if final_section:
            sections.append(final_section)

        # For each section, if it exceeds max_tokens, further split it into overlapping chunks.
        chunks = []
        current_chunk = ""
        for section in sections:
            token_count = TextChunker.count_tokens(section)
            current_token_count = TextChunker.count_tokens(current_chunk)
            if token_count + current_token_count < max_tokens + merge_margin:
                current_chunk += "\n" + section
            elif token_count > max_tokens + merge_margin:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                tokens = section.split()
                subchunks = TextChunker.chunk_text_with_overlap(tokens, max_tokens)
                chunks.extend(subchunks)
            else:
                chunks.append(current_chunk)
                current_chunk = ""
                chunks.append(section)
        return chunks