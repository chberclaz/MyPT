"""
Document loader for various file formats.

Supports:
- Plain text (.txt)
- Markdown (.md)
- Future: PDF, DOCX, etc.

Extensible via DocumentLoader.register_handler() for custom formats.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Iterator
from datetime import datetime


@dataclass
class Document:
    """
    Represents a loaded document with metadata.
    
    Attributes:
        text: Full document text content
        source: Source file path
        filename: Just the filename (for display)
        format: File format (txt, md, pdf, etc.)
        metadata: Additional metadata (modified time, size, etc.)
    """
    text: str
    source: str
    filename: str
    format: str
    metadata: Dict = field(default_factory=dict)
    
    @property
    def num_chars(self) -> int:
        return len(self.text)
    
    @property
    def num_lines(self) -> int:
        return self.text.count('\n') + 1


# Type for format handlers: (filepath) -> text
FormatHandler = Callable[[str], str]


class DocumentLoader:
    """
    Load documents from files with format-specific handling.
    
    Usage:
        loader = DocumentLoader()
        docs = loader.load_directory("workspace/docs")
        
        for doc in docs:
            print(f"{doc.filename}: {doc.num_chars} chars")
    """
    
    # Default supported extensions
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.text', '.markdown'}
    
    def __init__(self):
        self._handlers: Dict[str, FormatHandler] = {
            '.txt': self._load_text,
            '.text': self._load_text,
            '.md': self._load_markdown,
            '.markdown': self._load_markdown,
        }
    
    def register_handler(self, extension: str, handler: FormatHandler) -> None:
        """
        Register a custom handler for a file extension.
        
        Args:
            extension: File extension including dot (e.g., '.pdf')
            handler: Function that takes filepath and returns text
        """
        self._handlers[extension.lower()] = handler
    
    def _load_text(self, filepath: str) -> str:
        """Load plain text file."""
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    def _load_markdown(self, filepath: str) -> str:
        """Load markdown file (currently same as text, could strip formatting)."""
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    def _get_file_metadata(self, filepath: str) -> Dict:
        """Extract file metadata."""
        path = Path(filepath)
        stat = path.stat()
        return {
            'size_bytes': stat.st_size,
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        }
    
    def load_file(self, filepath: str) -> Optional[Document]:
        """
        Load a single document from file.
        
        Args:
            filepath: Path to file
            
        Returns:
            Document instance or None if format not supported
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        
        if ext not in self._handlers:
            return None
        
        try:
            text = self._handlers[ext](filepath)
            metadata = self._get_file_metadata(filepath)
            
            return Document(
                text=text,
                source=str(path.absolute()),
                filename=path.name,
                format=ext.lstrip('.'),
                metadata=metadata,
            )
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_directory(
        self, 
        directory: str, 
        recursive: bool = True,
        extensions: Optional[set] = None
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search subdirectories
            extensions: Specific extensions to load (default: all supported)
            
        Returns:
            List of Document instances
        """
        docs = []
        exts = extensions or set(self._handlers.keys())
        
        path = Path(directory)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        pattern = '**/*' if recursive else '*'
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in exts:
                doc = self.load_file(str(file_path))
                if doc:
                    docs.append(doc)
        
        return docs
    
    def iter_directory(
        self, 
        directory: str, 
        recursive: bool = True
    ) -> Iterator[Document]:
        """
        Iterate over documents in a directory (memory-efficient).
        
        Yields:
            Document instances one at a time
        """
        path = Path(directory)
        pattern = '**/*' if recursive else '*'
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self._handlers:
                doc = self.load_file(str(file_path))
                if doc:
                    yield doc

