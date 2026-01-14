#!/usr/bin/env python
"""
Extract documentation from source code files.

Supports:
- Go (.go) - Extracts doc comments
- TypeScript (.ts) - Extracts JSDoc comments
- Generic code with comments
"""

import re
from pathlib import Path
from typing import Optional, Tuple, List


class CodeDocExtractor:
    """Extract documentation comments from source code files."""
    
    # File extensions we handle
    SUPPORTED_EXTENSIONS = {'.go', '.ts', '.tsx'}
    
    # Go doc comment pattern (// comments before declarations)
    GO_DOC_PATTERN = re.compile(
        r'((?://[^\n]*\n)+)\s*'
        r'(?:func|type|var|const|package)\s+(\w+)',
        re.MULTILINE
    )
    
    # JSDoc/TSDoc pattern
    JSDOC_PATTERN = re.compile(
        r'/\*\*\s*(.*?)\s*\*/',
        re.DOTALL
    )
    
    def __init__(self):
        pass
    
    def is_supported(self, path: str) -> bool:
        """Check if file type is supported."""
        return Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def convert(self, path: str) -> Tuple[Optional[str], str]:
        """
        Extract documentation from a source file.
        
        Returns:
            Tuple of (content, status_message)
            content is None if extraction failed
        """
        path = Path(path)
        ext = path.suffix.lower()
        
        if not path.exists():
            return None, f"File not found: {path}"
        
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            if ext == '.go':
                result = self._extract_go_docs(content, path.stem)
            elif ext in ('.ts', '.tsx'):
                result = self._extract_ts_docs(content, path.stem)
            else:
                return None, "unsupported_extension"
            
            if result and len(result) >= 100:
                return result, f"code_doc_{ext[1:]}"
            else:
                return None, "no_docs_found"
                
        except Exception as e:
            return None, f"Error: {e}"
    
    def _extract_go_docs(self, content: str, filename: str) -> Optional[str]:
        """Extract Go documentation comments."""
        output_parts: List[str] = []
        
        # Get package name
        pkg_match = re.search(r'package\s+(\w+)', content)
        package = pkg_match.group(1) if pkg_match else filename
        
        # Find all doc comments with their declarations
        for match in self.GO_DOC_PATTERN.finditer(content):
            comments = match.group(1)
            name = match.group(2)
            
            # Clean up the comments (remove // prefixes)
            lines = []
            for line in comments.split('\n'):
                line = line.strip()
                if line.startswith('//'):
                    line = line[2:].strip()
                if line:
                    lines.append(line)
            
            if lines:
                doc = '\n'.join(lines)
                output_parts.append(f"## {name}\n\n{doc}")
        
        # Also extract standalone comment blocks
        block_comments = re.findall(r'/\*([^*]|\*[^/])*\*/', content)
        for comment in block_comments:
            # Clean up
            cleaned = re.sub(r'^\s*\*\s?', '', comment, flags=re.MULTILINE)
            cleaned = cleaned.strip()
            if len(cleaned) > 100:
                output_parts.append(cleaned)
        
        if not output_parts:
            return None
        
        header = f"# {filename}\nPackage: {package}\n"
        return header + "\n\n---\n\n".join(output_parts)
    
    def _extract_ts_docs(self, content: str, filename: str) -> Optional[str]:
        """Extract TypeScript/JSDoc documentation."""
        output_parts: List[str] = []
        
        # Find all JSDoc comments
        for match in self.JSDOC_PATTERN.finditer(content):
            doc_content = match.group(1)
            
            # Clean up
            lines = []
            for line in doc_content.split('\n'):
                line = re.sub(r'^\s*\*\s?', '', line)
                lines.append(line)
            
            cleaned = '\n'.join(lines).strip()
            
            if len(cleaned) > 50:
                # Process JSDoc tags
                cleaned = self._process_jsdoc_tags(cleaned)
                output_parts.append(cleaned)
        
        if not output_parts:
            return None
        
        header = f"# {filename}\n"
        return header + "\n\n---\n\n".join(output_parts)
    
    def _process_jsdoc_tags(self, text: str) -> str:
        """Process JSDoc tags."""
        # @param name description
        text = re.sub(r'@param\s+(\w+)\s+', r'\n- Parameter `\1`: ', text)
        text = re.sub(r'@param\s+\{[^}]+\}\s+(\w+)\s+', r'\n- Parameter `\1`: ', text)
        
        # @returns/@return
        text = re.sub(r'@returns?\s+', r'\n- Returns: ', text)
        
        # @throws
        text = re.sub(r'@throws\s+', r'\n- Throws: ', text)
        
        # @example
        text = re.sub(r'@example\s*\n?', r'\n\nExample:\n```\n', text)
        
        # @deprecated
        text = re.sub(r'@deprecated\s*', r'\n**DEPRECATED**: ', text)
        
        # @see
        text = re.sub(r'@see\s+', r'\n- See: ', text)
        
        # @since, @version
        text = re.sub(r'@since\s+', r'\n- Since: ', text)
        text = re.sub(r'@version\s+', r'\n- Version: ', text)
        
        # Remove other tags
        text = re.sub(r'@\w+\s*', '', text)
        
        return text.strip()


def convert_code(path: str) -> Tuple[Optional[str], str]:
    """Convenience function to convert a source code file."""
    extractor = CodeDocExtractor()
    return extractor.convert(path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python code_to_txt.py <source_file>")
        sys.exit(1)
    
    extractor = CodeDocExtractor()
    
    for path in sys.argv[1:]:
        content, status = extractor.convert(path)
        if content:
            print(f"=== {path} ({status}) ===")
            print(content[:2000])
            print(f"... ({len(content)} chars total)")
        else:
            print(f"Failed: {path}: {status}")

