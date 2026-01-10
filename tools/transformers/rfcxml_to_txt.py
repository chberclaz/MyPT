#!/usr/bin/env python
"""
Convert RFC XML documents to plaintext.

Handles RFC XML format from IETF, extracting:
- Title, abstract, and author information
- Section headings and content
- Preserving structure while removing XML markup
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, List


class RFCXMLConverter:
    """Convert RFC XML documents to clean plaintext."""
    
    # XML namespace commonly used in RFC XML
    NAMESPACES = {
        'rfc': 'http://www.rfc-editor.org/rfc-index',
        '': ''  # default namespace
    }
    
    def __init__(self):
        pass
    
    def is_rfc_xml(self, path: str) -> bool:
        """Check if a file appears to be RFC XML."""
        p = Path(path)
        if p.suffix.lower() != '.xml':
            return False
        
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                # Read first few lines to check for RFC indicators
                header = f.read(1000)
                return '<rfc' in header.lower() or 'rfc-editor' in header.lower()
        except:
            return False
    
    def convert(self, path: str) -> Tuple[Optional[str], str]:
        """
        Convert an RFC XML file to plaintext.
        
        Returns:
            Tuple of (content, status_message)
            content is None if conversion failed
        """
        path = Path(path)
        
        if not path.exists():
            return None, f"File not found: {path}"
        
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Try to parse as XML
            try:
                root = ET.fromstring(content)
                return self._convert_xml(root), "xml_parser"
            except ET.ParseError:
                # Fallback to regex-based extraction
                return self._extract_text_fallback(content), "regex_fallback"
                
        except Exception as e:
            return None, f"Error: {e}"
    
    def _convert_xml(self, root: ET.Element) -> str:
        """Convert parsed XML tree to plaintext."""
        output_lines: List[str] = []
        
        # Extract RFC number/title from attributes or front matter
        rfc_number = root.get('number', '')
        if rfc_number:
            output_lines.append(f"RFC {rfc_number}")
            output_lines.append('')
        
        # Process front matter
        front = root.find('.//front')
        if front is not None:
            self._process_front(front, output_lines)
        
        # Process middle (main content)
        middle = root.find('.//middle')
        if middle is not None:
            self._process_sections(middle, output_lines, level=1)
        
        # Process back matter (references, appendices)
        back = root.find('.//back')
        if back is not None:
            output_lines.append('')
            output_lines.append('REFERENCES AND APPENDICES')
            output_lines.append('=' * 40)
            self._process_sections(back, output_lines, level=1)
        
        return '\n'.join(output_lines)
    
    def _process_front(self, front: ET.Element, output: List[str]) -> None:
        """Process front matter (title, abstract, authors)."""
        # Title
        title = front.find('.//title')
        if title is not None and title.text:
            output.append(title.text.strip())
            output.append('=' * min(70, len(title.text.strip())))
            output.append('')
        
        # Authors
        authors = front.findall('.//author')
        if authors:
            author_names = []
            for author in authors:
                name = author.get('fullname', '')
                if not name:
                    # Try to construct from parts
                    first = author.get('initials', '') or author.get('firstname', '')
                    last = author.get('surname', '') or author.get('lastname', '')
                    name = f"{first} {last}".strip()
                if name:
                    author_names.append(name)
            
            if author_names:
                output.append(f"Authors: {', '.join(author_names)}")
                output.append('')
        
        # Abstract
        abstract = front.find('.//abstract')
        if abstract is not None:
            output.append('ABSTRACT')
            output.append('-' * 40)
            output.append(self._get_text_content(abstract))
            output.append('')
    
    def _process_sections(self, element: ET.Element, output: List[str], level: int = 1) -> None:
        """Recursively process sections."""
        for child in element:
            tag = child.tag.split('}')[-1]  # Remove namespace
            
            if tag == 'section':
                self._process_section(child, output, level)
            elif tag == 'references':
                self._process_references(child, output)
            elif tag in ('t', 'p'):  # Text paragraphs
                text = self._get_text_content(child)
                if text:
                    output.append(text)
                    output.append('')
            elif tag == 'figure':
                self._process_figure(child, output)
            elif tag == 'ul' or tag == 'ol':
                self._process_list(child, output, ordered=(tag == 'ol'))
            elif tag == 'dl':
                self._process_definition_list(child, output)
            elif tag == 'artwork' or tag == 'sourcecode':
                self._process_code(child, output)
    
    def _process_section(self, section: ET.Element, output: List[str], level: int) -> None:
        """Process a single section."""
        title = section.get('title', '') or section.get('name', '')
        if not title:
            # Try to find title element
            title_el = section.find('./name')
            if title_el is not None:
                title = self._get_text_content(title_el)
        
        if title:
            output.append('')
            if level == 1:
                output.append(title.upper())
                output.append('=' * min(60, len(title)))
            elif level == 2:
                output.append(title)
                output.append('-' * min(50, len(title)))
            else:
                output.append(f"{'#' * level} {title}")
            output.append('')
        
        self._process_sections(section, output, level + 1)
    
    def _process_figure(self, figure: ET.Element, output: List[str]) -> None:
        """Process figure elements."""
        name = figure.get('title', '') or figure.get('name', '')
        if name:
            output.append(f"[Figure: {name}]")
        
        # Process artwork/sourcecode within figure
        for child in figure:
            tag = child.tag.split('}')[-1]
            if tag in ('artwork', 'sourcecode'):
                self._process_code(child, output)
    
    def _process_code(self, element: ET.Element, output: List[str]) -> None:
        """Process code/artwork blocks."""
        code_type = element.get('type', 'code')
        text = element.text or ''
        
        if text.strip():
            output.append('')
            output.append(f'[CODE: {code_type}]')
            for line in text.split('\n'):
                output.append(f'  {line}')
            output.append('[/CODE]')
            output.append('')
    
    def _process_list(self, element: ET.Element, output: List[str], ordered: bool = False) -> None:
        """Process ordered/unordered lists."""
        for i, item in enumerate(element.findall('./li'), 1):
            marker = f"{i}." if ordered else "-"
            text = self._get_text_content(item)
            output.append(f"  {marker} {text}")
        output.append('')
    
    def _process_definition_list(self, element: ET.Element, output: List[str]) -> None:
        """Process definition lists."""
        for dt in element.findall('./dt'):
            term = self._get_text_content(dt)
            output.append(f"  {term}:")
        for dd in element.findall('./dd'):
            definition = self._get_text_content(dd)
            output.append(f"    {definition}")
        output.append('')
    
    def _process_references(self, refs: ET.Element, output: List[str]) -> None:
        """Process references section."""
        title = refs.get('title', 'References')
        output.append('')
        output.append(title.upper())
        output.append('-' * 40)
        
        for ref in refs.findall('.//reference'):
            ref_id = ref.get('anchor', '')
            
            # Try to get reference title
            title_el = ref.find('.//title')
            title = self._get_text_content(title_el) if title_el is not None else ''
            
            if ref_id and title:
                output.append(f"  [{ref_id}] {title}")
        
        output.append('')
    
    def _get_text_content(self, element: ET.Element) -> str:
        """Extract all text content from an element and its children."""
        if element is None:
            return ''
        
        texts = []
        
        # Get direct text
        if element.text:
            texts.append(element.text.strip())
        
        # Get text from children
        for child in element:
            child_text = self._get_text_content(child)
            if child_text:
                texts.append(child_text)
            
            # Get tail text
            if child.tail:
                texts.append(child.tail.strip())
        
        return ' '.join(texts)
    
    def _extract_text_fallback(self, content: str) -> str:
        """Fallback regex-based text extraction."""
        output_lines: List[str] = []
        
        # Remove XML declaration and comments
        content = re.sub(r'<\?xml[^>]*\?>', '', content)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Extract title
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        if title_match:
            output_lines.append(title_match.group(1).strip())
            output_lines.append('=' * 60)
            output_lines.append('')
        
        # Extract abstract
        abstract_match = re.search(r'<abstract[^>]*>(.*?)</abstract>', content, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            output_lines.append('ABSTRACT')
            output_lines.append('-' * 40)
            abstract_text = re.sub(r'<[^>]+>', ' ', abstract_match.group(1))
            abstract_text = ' '.join(abstract_text.split())
            output_lines.append(abstract_text)
            output_lines.append('')
        
        # Extract section titles and content
        sections = re.findall(r'<section[^>]*title=["\']([^"\']+)["\'][^>]*>(.*?)</section>', 
                             content, re.DOTALL | re.IGNORECASE)
        
        for title, section_content in sections:
            output_lines.append('')
            output_lines.append(title.upper())
            output_lines.append('=' * min(60, len(title)))
            
            # Extract text from paragraphs
            paragraphs = re.findall(r'<t[^>]*>(.*?)</t>', section_content, re.DOTALL)
            for para in paragraphs:
                para_text = re.sub(r'<[^>]+>', ' ', para)
                para_text = ' '.join(para_text.split())
                if para_text:
                    output_lines.append(para_text)
                    output_lines.append('')
        
        return '\n'.join(output_lines)


def convert_rfc_xml(path: str) -> Tuple[Optional[str], str]:
    """Convenience function to convert a single RFC XML file."""
    converter = RFCXMLConverter()
    return converter.convert(path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rfcxml_to_txt.py <rfc_xml_file>")
        sys.exit(1)
    
    converter = RFCXMLConverter()
    
    for path in sys.argv[1:]:
        content, status = converter.convert(path)
        if content:
            print(f"=== {path} ({status}) ===")
            print(content[:3000])
            print(f"... ({len(content)} chars total)")
        else:
            print(f"Failed to convert {path}: {status}")

