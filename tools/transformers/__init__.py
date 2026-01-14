"""
Text transformers for the Phase 2 domain corpus builder.

This package contains converters for various document formats:
- man_to_txt: Unix man pages
- rfcxml_to_txt: RFC XML format
- html_to_txt: HTML documents
- md_rst_to_txt: Markdown and reStructuredText
- java_to_txt: Java source files (Javadoc extraction)
- texinfo_to_txt: GNU Texinfo format
- rfc_downloader: Download actual RFCs from rfc-editor.org
"""

from .man_to_txt import ManPageConverter, convert_manpage
from .rfcxml_to_txt import RFCXMLConverter, convert_rfc_xml
from .html_to_txt import HTMLConverter, convert_html
from .md_rst_to_txt import MarkdownRSTConverter, convert_md_rst
from .java_to_txt import JavaDocConverter, convert_java
from .texinfo_to_txt import TexinfoConverter, convert_texinfo
from .code_to_txt import CodeDocExtractor, convert_code
from .rfc_downloader import RFCDownloader, download_rfcs

__all__ = [
    'ManPageConverter', 'convert_manpage',
    'RFCXMLConverter', 'convert_rfc_xml',
    'HTMLConverter', 'convert_html',
    'MarkdownRSTConverter', 'convert_md_rst',
    'JavaDocConverter', 'convert_java',
    'TexinfoConverter', 'convert_texinfo',
    'CodeDocExtractor', 'convert_code',
    'RFCDownloader', 'download_rfcs',
]

