"""
Text chunking for RAG indexing.

Splits documents into overlapping chunks for embedding.
Preserves source information for provenance tracking.
"""

from dataclasses import dataclass
from typing import List, Optional, Iterator


@dataclass
class Chunk:
    """
    A text chunk with full provenance information.
    
    Attributes:
        chunk_id: Unique identifier within the index
        text: Chunk text content
        source: Source information dict with file, start_char, end_char, etc.
    """
    chunk_id: int
    text: str
    source: dict
    
    @property
    def num_chars(self) -> int:
        return len(self.text)
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
        }


class TextChunker:
    """
    Split text into overlapping chunks.
    
    Args:
        chunk_size: Target chunk size in characters (default: 1200)
        chunk_overlap: Overlap between chunks in characters (default: 200)
        min_chunk_size: Minimum chunk size to emit (default: 50)
    
    Usage:
        chunker = TextChunker(chunk_size=1200, chunk_overlap=200)
        chunks = chunker.chunk_document(document)
    """
    
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Validate
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def _find_break_point(self, text: str, target_pos: int, window: int = 100) -> int:
        """
        Find a good break point near target_pos (paragraph, sentence, or word boundary).
        
        Args:
            text: Full text
            target_pos: Target position
            window: Search window size
            
        Returns:
            Best break position
        """
        start = max(0, target_pos - window)
        end = min(len(text), target_pos + window)
        search_region = text[start:end]
        
        # Priority: paragraph > sentence > word
        # Look for paragraph break
        for sep in ['\n\n', '\n']:
            idx = search_region.rfind(sep, 0, target_pos - start + window // 2)
            if idx != -1:
                return start + idx + len(sep)
        
        # Look for sentence end
        for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            idx = search_region.rfind(sep, 0, target_pos - start + window // 2)
            if idx != -1:
                return start + idx + len(sep)
        
        # Look for word boundary
        idx = search_region.rfind(' ', 0, target_pos - start + window // 2)
        if idx != -1:
            return start + idx + 1
        
        # Fall back to target position
        return target_pos
    
    def chunk_text(
        self,
        text: str,
        source_file: str = "",
        source_filename: str = "",
        start_chunk_id: int = 0,
    ) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            source_file: Full source file path
            source_filename: Just filename (for display)
            start_chunk_id: Starting chunk ID
            
        Returns:
            List of Chunk instances
        """
        if not text or len(text) < self.min_chunk_size:
            if text:
                return [Chunk(
                    chunk_id=start_chunk_id,
                    text=text,
                    source={
                        "file": source_file,
                        "filename": source_filename,
                        "start_char": 0,
                        "end_char": len(text),
                        "start_line": 1,
                        "end_line": text.count('\n') + 1,
                    }
                )]
            return []
        
        chunks = []
        chunk_id = start_chunk_id
        pos = 0
        
        while pos < len(text):
            # Determine end position
            end_pos = pos + self.chunk_size
            
            if end_pos >= len(text):
                # Last chunk
                chunk_text = text[pos:]
            else:
                # Find good break point
                end_pos = self._find_break_point(text, end_pos)
                chunk_text = text[pos:end_pos]
            
            # Skip if too small (unless it's the last chunk)
            if len(chunk_text) >= self.min_chunk_size or pos + len(chunk_text) >= len(text):
                # Calculate line numbers
                start_line = text[:pos].count('\n') + 1
                end_line = start_line + chunk_text.count('\n')
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    source={
                        "file": source_file,
                        "filename": source_filename,
                        "start_char": pos,
                        "end_char": pos + len(chunk_text),
                        "start_line": start_line,
                        "end_line": end_line,
                    }
                ))
                chunk_id += 1
            
            # Move position with overlap
            step = self.chunk_size - self.chunk_overlap
            pos += step
            
            # Avoid infinite loop
            if step <= 0:
                break
        
        return chunks
    
    def chunk_document(self, document, start_chunk_id: int = 0) -> List[Chunk]:
        """
        Chunk a Document instance.
        
        Args:
            document: Document instance from loader
            start_chunk_id: Starting chunk ID
            
        Returns:
            List of Chunk instances
        """
        return self.chunk_text(
            text=document.text,
            source_file=document.source,
            source_filename=document.filename,
            start_chunk_id=start_chunk_id,
        )
    
    def chunk_documents(self, documents: list) -> List[Chunk]:
        """
        Chunk multiple documents with continuous chunk IDs.
        
        Args:
            documents: List of Document instances
            
        Returns:
            List of all Chunk instances
        """
        all_chunks = []
        chunk_id = 0
        
        for doc in documents:
            chunks = self.chunk_document(doc, start_chunk_id=chunk_id)
            all_chunks.extend(chunks)
            chunk_id += len(chunks)
        
        return all_chunks
    
    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
        }

