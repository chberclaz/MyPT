"""
Transformers for converting various document formats to plaintext.

Available transformers:
- man_to_txt: Convert Unix man pages to plaintext
- rfcxml_to_txt: Convert RFC XML documents to plaintext
- html_to_txt: Convert HTML documents to plaintext
- md_rst_to_txt: Convert Markdown and RST documents to plaintext
"""

from .man_to_txt import ManPageConverter
from .rfcxml_to_txt import RFCXMLConverter
from .html_to_txt import HTMLConverter
from .md_rst_to_txt import MarkdownRSTConverter

__all__ = [
    'ManPageConverter',
    'RFCXMLConverter', 
    'HTMLConverter',
    'MarkdownRSTConverter',
]

