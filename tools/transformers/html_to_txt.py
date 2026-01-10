#!/usr/bin/env python
"""
Convert HTML documents to plaintext.

Designed for documentation sites like MDN, NIST, etc.
- Removes navigation, footers, sidebars
- Preserves headings, code blocks, and lists
- Handles common doc site patterns
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Tuple, List, Set


class HTMLToTextParser(HTMLParser):
    """HTML parser that extracts clean text content."""
    
    # Tags to completely skip (navigation, footers, etc.)
    SKIP_TAGS = {
        'nav', 'footer', 'aside', 'header', 'script', 'style', 'noscript',
        'svg', 'iframe', 'form', 'button', 'input', 'select', 'textarea',
        'menu', 'menuitem'
    }
    
    # Block-level tags that should have surrounding newlines
    BLOCK_TAGS = {
        'p', 'div', 'section', 'article', 'main', 'blockquote',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'dl', 'li', 'dt', 'dd',
        'pre', 'code', 'table', 'tr', 'th', 'td',
        'figure', 'figcaption', 'details', 'summary'
    }
    
    # Heading tags
    HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    
    # Code-related tags
    CODE_TAGS = {'pre', 'code'}
    
    # Classes/IDs that indicate navigation/boilerplate
    SKIP_CLASSES = {
        'nav', 'navigation', 'sidebar', 'footer', 'header', 'menu',
        'breadcrumb', 'breadcrumbs', 'toc', 'table-of-contents',
        'edit-link', 'edit-page', 'share', 'social', 'cookie',
        'banner', 'alert', 'ad', 'advertisement', 'promo'
    }
    
    def __init__(self):
        super().__init__()
        self.output: List[str] = []
        self.current_text: List[str] = []
        self.skip_depth = 0
        self.in_code = False
        self.in_heading = False
        self.heading_level = 0
        self.list_stack: List[str] = []  # 'ul' or 'ol'
        self.list_counters: List[int] = []
    
    def _should_skip(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> bool:
        """Check if this tag should be skipped based on tag name or attributes."""
        if tag in self.SKIP_TAGS:
            return True
        
        # Check class and id attributes for skip patterns
        for attr, value in attrs:
            if attr in ('class', 'id') and value:
                value_lower = value.lower()
                for skip_class in self.SKIP_CLASSES:
                    if skip_class in value_lower:
                        return True
        
        return False
    
    def _flush_text(self, add_newline: bool = False):
        """Flush accumulated text to output."""
        if self.current_text:
            text = ''.join(self.current_text).strip()
            if text:
                self.output.append(text)
            self.current_text = []
        
        if add_newline:
            self.output.append('')
    
    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        if self.skip_depth > 0:
            self.skip_depth += 1
            return
        
        if self._should_skip(tag, attrs):
            self.skip_depth = 1
            return
        
        tag = tag.lower()
        
        if tag in self.BLOCK_TAGS:
            self._flush_text(add_newline=True)
        
        if tag in self.HEADING_TAGS:
            self.in_heading = True
            self.heading_level = int(tag[1])
        
        if tag in self.CODE_TAGS:
            if tag == 'pre' or (tag == 'code' and not self.in_code):
                self._flush_text()
                self.output.append('[CODE]')
                self.in_code = True
        
        if tag == 'ul':
            self.list_stack.append('ul')
            self.list_counters.append(0)
        elif tag == 'ol':
            self.list_stack.append('ol')
            self.list_counters.append(0)
        elif tag == 'li':
            self._flush_text()
            indent = '  ' * (len(self.list_stack) - 1)
            if self.list_stack and self.list_stack[-1] == 'ol':
                self.list_counters[-1] += 1
                self.current_text.append(f"{indent}{self.list_counters[-1]}. ")
            else:
                self.current_text.append(f"{indent}- ")
        
        if tag == 'br':
            self._flush_text()
    
    def handle_endtag(self, tag: str):
        if self.skip_depth > 0:
            self.skip_depth -= 1
            return
        
        tag = tag.lower()
        
        if tag in self.HEADING_TAGS and self.in_heading:
            self._flush_text()
            # Add underline for top-level headings
            if self.output and self.heading_level <= 2:
                header_text = self.output[-1] if self.output else ''
                if self.heading_level == 1:
                    self.output.append('=' * min(60, len(header_text)))
                else:
                    self.output.append('-' * min(50, len(header_text)))
            self.output.append('')
            self.in_heading = False
            self.heading_level = 0
        
        if tag in self.CODE_TAGS:
            if tag == 'pre' or (tag == 'code' and self.in_code):
                self._flush_text()
                self.output.append('[/CODE]')
                self.output.append('')
                self.in_code = False
        
        if tag == 'ul' or tag == 'ol':
            if self.list_stack:
                self.list_stack.pop()
            if self.list_counters:
                self.list_counters.pop()
            self._flush_text(add_newline=True)
        
        if tag in self.BLOCK_TAGS:
            self._flush_text(add_newline=True)
    
    def handle_data(self, data: str):
        if self.skip_depth > 0:
            return
        
        if self.in_code:
            # Preserve whitespace in code blocks
            self.current_text.append(data)
        else:
            # Normalize whitespace for regular text
            text = ' '.join(data.split())
            if text:
                self.current_text.append(text + ' ')
    
    def handle_entityref(self, name: str):
        if self.skip_depth > 0:
            return
        
        # Common HTML entities
        entities = {
            'nbsp': ' ',
            'lt': '<',
            'gt': '>',
            'amp': '&',
            'quot': '"',
            'apos': "'",
            'mdash': '—',
            'ndash': '–',
            'hellip': '...',
            'copy': '©',
            'reg': '®',
            'trade': '™',
        }
        self.current_text.append(entities.get(name, f'&{name};'))
    
    def handle_charref(self, name: str):
        if self.skip_depth > 0:
            return
        
        try:
            if name.startswith('x'):
                char = chr(int(name[1:], 16))
            else:
                char = chr(int(name))
            self.current_text.append(char)
        except (ValueError, OverflowError):
            self.current_text.append(f'&#{name};')
    
    def get_text(self) -> str:
        """Get the final text output."""
        self._flush_text()
        
        # Clean up output
        lines = []
        prev_empty = False
        
        for line in self.output:
            is_empty = not line.strip()
            
            # Skip consecutive empty lines
            if is_empty and prev_empty:
                continue
            
            lines.append(line)
            prev_empty = is_empty
        
        # Remove leading/trailing empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)


class HTMLConverter:
    """Convert HTML documents to clean plaintext."""
    
    def __init__(self):
        pass
    
    def is_html(self, path: str) -> bool:
        """Check if a file is HTML."""
        p = Path(path)
        return p.suffix.lower() in {'.html', '.htm', '.xhtml'}
    
    def convert(self, path: str) -> Tuple[Optional[str], str]:
        """
        Convert an HTML file to plaintext.
        
        Returns:
            Tuple of (content, status_message)
            content is None if conversion failed
        """
        path = Path(path)
        
        if not path.exists():
            return None, f"File not found: {path}"
        
        try:
            # Try different encodings
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return None, "Could not decode file"
            
            # Parse and convert
            parser = HTMLToTextParser()
            parser.feed(content)
            
            result = parser.get_text()
            result = self._post_process(result)
            
            return result, "html_parser"
            
        except Exception as e:
            return None, f"Error: {e}"
    
    def convert_string(self, html: str) -> str:
        """Convert an HTML string to plaintext."""
        parser = HTMLToTextParser()
        parser.feed(html)
        result = parser.get_text()
        return self._post_process(result)
    
    def _post_process(self, text: str) -> str:
        """Post-process converted text."""
        # Remove common boilerplate patterns
        boilerplate_patterns = [
            r'Edit this page on GitHub.*',
            r'Previous\s*[|/]\s*Next',
            r'Table of Contents',
            r'Skip to .*',
            r'© \d{4}.*',
            r'All rights reserved.*',
            r'Terms of Service.*',
            r'Privacy Policy.*',
            r'Cookie.*preferences.*',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up excessive newlines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Remove lines that are just dashes/equals (orphaned underlines)
        lines = text.split('\n')
        cleaned_lines = []
        for i, line in enumerate(lines):
            if re.match(r'^[-=]+$', line.strip()):
                # Keep underlines only if previous line has content
                if i > 0 and cleaned_lines and cleaned_lines[-1].strip():
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()


def convert_html(path: str) -> Tuple[Optional[str], str]:
    """Convenience function to convert a single HTML file."""
    converter = HTMLConverter()
    return converter.convert(path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python html_to_txt.py <html_file>")
        sys.exit(1)
    
    converter = HTMLConverter()
    
    for path in sys.argv[1:]:
        content, status = converter.convert(path)
        if content:
            print(f"=== {path} ({status}) ===")
            print(content[:3000])
            print(f"... ({len(content)} chars total)")
        else:
            print(f"Failed to convert {path}: {status}")

