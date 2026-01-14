#!/usr/bin/env python
"""
Extract documentation from Java source files.

Extracts:
- Javadoc comments (/** ... */)
- Class/interface/method signatures with their docs
- Package documentation
"""

import re
from pathlib import Path
from typing import Optional, Tuple, List


class JavaDocConverter:
    """Extract Javadoc documentation from Java source files."""
    
    # Regex patterns
    JAVADOC_PATTERN = re.compile(
        r'/\*\*\s*(.*?)\s*\*/',
        re.DOTALL
    )
    
    CLASS_PATTERN = re.compile(
        r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?'
        r'(?:class|interface|enum|record)\s+(\w+)',
        re.MULTILINE
    )
    
    METHOD_PATTERN = re.compile(
        r'(?:public|protected|private)?\s*'
        r'(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?'
        r'(?:abstract\s+)?(?:native\s+)?'
        r'(?:<[^>]+>\s+)?'  # Generic type params
        r'(\w+(?:\[\])?(?:<[^>]+>)?)\s+'  # Return type
        r'(\w+)\s*\([^)]*\)',  # Method name and params
        re.MULTILINE
    )
    
    def __init__(self):
        pass
    
    def is_java_file(self, path: str) -> bool:
        """Check if file is a Java source file."""
        return Path(path).suffix.lower() == '.java'
    
    def convert(self, path: str) -> Tuple[Optional[str], str]:
        """
        Extract documentation from a Java file.
        
        Returns:
            Tuple of (content, status_message)
            content is None if extraction failed
        """
        path = Path(path)
        
        if not path.exists():
            return None, f"File not found: {path}"
        
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            docs = self._extract_documentation(content, path.stem)
            
            if docs:
                return docs, "javadoc"
            else:
                return None, "no_javadoc"
                
        except Exception as e:
            return None, f"Error: {e}"
    
    def _extract_documentation(self, content: str, class_name: str) -> Optional[str]:
        """Extract all documentation from Java source."""
        output_parts: List[str] = []
        
        # Find package
        package_match = re.search(r'package\s+([\w.]+)\s*;', content)
        package = package_match.group(1) if package_match else ""
        
        # Find all Javadoc comments with their following declarations
        # Split content into segments around Javadoc comments
        segments = self._split_by_javadoc(content)
        
        for javadoc, following_code in segments:
            if not javadoc:
                continue
            
            # Clean the Javadoc
            clean_doc = self._clean_javadoc(javadoc)
            
            if not clean_doc or len(clean_doc) < 20:
                continue
            
            # Try to find what this documents
            declaration = self._find_declaration(following_code)
            
            if declaration:
                output_parts.append(f"## {declaration}\n\n{clean_doc}")
            else:
                output_parts.append(clean_doc)
        
        if not output_parts:
            return None
        
        # Add header
        header = f"# {class_name}"
        if package:
            header += f"\nPackage: {package}"
        
        return header + "\n\n" + "\n\n---\n\n".join(output_parts)
    
    def _split_by_javadoc(self, content: str) -> List[Tuple[str, str]]:
        """Split content into (javadoc, following_code) pairs."""
        results = []
        
        # Find all Javadoc comments
        for match in self.JAVADOC_PATTERN.finditer(content):
            javadoc = match.group(1)
            # Get the code following this Javadoc (up to 500 chars or next Javadoc)
            start = match.end()
            end = min(start + 500, len(content))
            
            # Find next Javadoc or class/method boundary
            next_javadoc = content.find('/**', start)
            if next_javadoc != -1 and next_javadoc < end:
                end = next_javadoc
            
            following = content[start:end]
            results.append((javadoc, following))
        
        return results
    
    def _clean_javadoc(self, javadoc: str) -> str:
        """Clean Javadoc comment text."""
        lines = javadoc.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading asterisks and whitespace
            line = re.sub(r'^\s*\*\s?', '', line)
            line = line.rstrip()
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Process Javadoc tags
        text = self._process_javadoc_tags(text)
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _process_javadoc_tags(self, text: str) -> str:
        """Process Javadoc tags into readable format."""
        # @param name description -> Parameter name: description
        text = re.sub(
            r'@param\s+(\w+)\s+',
            r'\n- Parameter `\1`: ',
            text
        )
        
        # @return description -> Returns: description
        text = re.sub(r'@return\s+', r'\n- Returns: ', text)
        
        # @throws/@exception Type description
        text = re.sub(
            r'@(?:throws|exception)\s+(\w+)\s+',
            r'\n- Throws `\1`: ',
            text
        )
        
        # @see reference -> See: reference
        text = re.sub(r'@see\s+', r'\n- See: ', text)
        
        # @since version -> Since: version
        text = re.sub(r'@since\s+', r'\n- Since: ', text)
        
        # @deprecated message -> DEPRECATED: message
        text = re.sub(r'@deprecated\s*', r'\n**DEPRECATED**: ', text)
        
        # @author, @version - keep simple
        text = re.sub(r'@author\s+', r'\n- Author: ', text)
        text = re.sub(r'@version\s+', r'\n- Version: ', text)
        
        # {@code text} -> `text`
        text = re.sub(r'\{@code\s+([^}]+)\}', r'`\1`', text)
        
        # {@link Class#method} -> Class.method
        text = re.sub(r'\{@link\s+([^}]+)\}', r'`\1`', text)
        
        # {@literal text} -> text
        text = re.sub(r'\{@literal\s+([^}]+)\}', r'\1', text)
        
        # {@value} -> (value)
        text = re.sub(r'\{@value[^}]*\}', r'(value)', text)
        
        # Remove other @ tags we don't handle
        text = re.sub(r'@\w+\s*', '', text)
        
        # Clean up HTML tags commonly found in Javadoc
        text = re.sub(r'<p>\s*', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<br\s*/?\s*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</?code>', '`', text, flags=re.IGNORECASE)
        text = re.sub(r'</?pre>', '\n```\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</?[a-z][^>]*>', '', text, flags=re.IGNORECASE)
        
        return text
    
    def _find_declaration(self, code: str) -> Optional[str]:
        """Find the declaration following a Javadoc comment."""
        # Look for class/interface/enum
        class_match = self.CLASS_PATTERN.search(code)
        if class_match:
            return f"Class {class_match.group(1)}"
        
        # Look for method
        method_match = self.METHOD_PATTERN.search(code)
        if method_match:
            return_type = method_match.group(1)
            method_name = method_match.group(2)
            return f"{return_type} {method_name}()"
        
        # Look for field
        field_match = re.search(
            r'(?:public|protected|private)\s+(?:static\s+)?(?:final\s+)?'
            r'(\w+(?:<[^>]+>)?)\s+(\w+)\s*[;=]',
            code
        )
        if field_match:
            return f"Field {field_match.group(2)}"
        
        return None


def convert_java(path: str) -> Tuple[Optional[str], str]:
    """Convenience function to convert a single Java file."""
    converter = JavaDocConverter()
    return converter.convert(path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python java_to_txt.py <java_file>")
        sys.exit(1)
    
    converter = JavaDocConverter()
    
    for path in sys.argv[1:]:
        content, status = converter.convert(path)
        if content:
            print(f"=== {path} ({status}) ===")
            print(content[:3000])
            print(f"... ({len(content)} chars total)")
        else:
            print(f"Failed to convert {path}: {status}")

