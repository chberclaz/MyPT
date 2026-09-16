#!/usr/bin/env python
"""
Convert Markdown and RST (reStructuredText) documents to plaintext.

Preserves:
- Headings (converted to plain text with underlines)
- Code blocks (marked with [CODE]/[/CODE])
- Lists
- Basic formatting removal
"""

import re
from pathlib import Path
from typing import Optional, Tuple, List


class MarkdownRSTConverter:
    """Convert Markdown and RST documents to clean plaintext."""
    
    MARKDOWN_EXTENSIONS = {'.md', '.markdown', '.mdown', '.mkd'}
    RST_EXTENSIONS = {'.rst', '.rest', '.txt'}  # .txt might be RST
    
    def __init__(self):
        pass
    
    def is_markdown(self, path: str) -> bool:
        """Check if a file is Markdown."""
        return Path(path).suffix.lower() in self.MARKDOWN_EXTENSIONS
    
    def is_rst(self, path: str) -> bool:
        """Check if a file appears to be RST."""
        p = Path(path)
        if p.suffix.lower() in {'.rst', '.rest'}:
            return True
        
        # For .txt files, check content for RST patterns
        if p.suffix.lower() == '.txt':
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    header = f.read(2000)
                # RST indicators
                if re.search(r'^\.\. ', header, re.MULTILINE):  # RST directives
                    return True
                if re.search(r'^::\s*$', header, re.MULTILINE):  # Literal blocks
                    return True
                if re.search(r'``.+``', header):  # RST inline literals
                    return True
            except:
                pass
        
        return False
    
    def convert(self, path: str) -> Tuple[Optional[str], str]:
        """
        Convert a Markdown or RST file to plaintext.
        
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
            
            if self.is_markdown(str(path)):
                return self._convert_markdown(content), "markdown"
            elif self.is_rst(str(path)):
                return self._convert_rst(content), "rst"
            else:
                # Default to markdown conversion
                return self._convert_markdown(content), "markdown_default"
                
        except Exception as e:
            return None, f"Error: {e}"
    
    def _convert_markdown(self, content: str) -> str:
        """Convert Markdown to plaintext."""
        lines = content.split('\n')
        output_lines: List[str] = []
        in_code_block = False
        code_fence = ''
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Handle fenced code blocks
            fence_match = re.match(r'^(`{3,}|~{3,})(\w*)', line)
            if fence_match:
                if not in_code_block:
                    in_code_block = True
                    code_fence = fence_match.group(1)[0]  # ` or ~
                    lang = fence_match.group(2) or 'code'
                    output_lines.append('')
                    output_lines.append(f'[CODE: {lang}]')
                elif line.startswith(code_fence):
                    in_code_block = False
                    code_fence = ''
                    output_lines.append('[/CODE]')
                    output_lines.append('')
                else:
                    output_lines.append(line)
                i += 1
                continue
            
            if in_code_block:
                output_lines.append(line)
                i += 1
                continue
            
            # Handle indented code blocks (4 spaces or tab)
            if re.match(r'^(    |\t)', line) and not line.strip().startswith('-'):
                if output_lines and output_lines[-1] != '[CODE]':
                    output_lines.append('[CODE]')
                output_lines.append(line[4:] if line.startswith('    ') else line[1:])
                i += 1
                continue
            elif output_lines and output_lines[-1] != '[/CODE]' and len(output_lines) > 1:
                # Check if we were in an indented code block
                prev_idx = len(output_lines) - 1
                while prev_idx >= 0 and output_lines[prev_idx] == '':
                    prev_idx -= 1
                if prev_idx > 0 and output_lines[prev_idx - 1] == '[CODE]':
                    # Find where [CODE] started and check if this ends it
                    pass  # Complex case, skip for simplicity
            
            # ATX-style headings (# Heading)
            heading_match = re.match(r'^(#{1,6})\s+(.+?)\s*#*\s*$', line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                output_lines.append('')
                output_lines.append(title)
                if level == 1:
                    output_lines.append('=' * min(60, len(title)))
                elif level == 2:
                    output_lines.append('-' * min(50, len(title)))
                output_lines.append('')
                i += 1
                continue
            
            # Setext-style headings (underlined with = or -)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r'^=+\s*$', next_line) and line.strip():
                    output_lines.append('')
                    output_lines.append(line.strip())
                    output_lines.append('=' * min(60, len(line.strip())))
                    output_lines.append('')
                    i += 2
                    continue
                elif re.match(r'^-+\s*$', next_line) and line.strip() and not line.startswith('-'):
                    output_lines.append('')
                    output_lines.append(line.strip())
                    output_lines.append('-' * min(50, len(line.strip())))
                    output_lines.append('')
                    i += 2
                    continue
            
            # Process inline formatting
            processed = self._process_markdown_inline(line)
            output_lines.append(processed)
            i += 1
        
        return self._clean_output('\n'.join(output_lines))
    
    def _process_markdown_inline(self, line: str) -> str:
        """Process inline Markdown formatting."""
        # Remove images but keep alt text
        line = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[Image: \1]', line)
        
        # Convert links to just text
        line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
        
        # Remove reference-style link definitions
        line = re.sub(r'^\s*\[[^\]]+\]:\s*\S+.*$', '', line)
        
        # Remove bold/italic markers
        line = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', line)
        line = re.sub(r'___(.+?)___', r'\1', line)
        line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
        line = re.sub(r'__(.+?)__', r'\1', line)
        line = re.sub(r'\*(.+?)\*', r'\1', line)
        line = re.sub(r'_(.+?)_', r'\1', line)
        
        # Keep inline code as-is
        line = re.sub(r'`([^`]+)`', r'`\1`', line)
        
        # Convert horizontal rules
        if re.match(r'^[-*_]{3,}\s*$', line):
            line = '-' * 40
        
        return line
    
    def _convert_rst(self, content: str) -> str:
        """Convert RST to plaintext."""
        lines = content.split('\n')
        output_lines: List[str] = []
        in_code_block = False
        in_directive = False
        directive_indent = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Handle code block directive
            if re.match(r'^\.\.\s+code(-block)?::', line):
                lang_match = re.search(r'::\s*(\w+)?', line)
                lang = lang_match.group(1) if lang_match and lang_match.group(1) else 'code'
                output_lines.append('')
                output_lines.append(f'[CODE: {lang}]')
                in_code_block = True
                directive_indent = len(line) - len(line.lstrip()) + 3
                i += 1
                continue
            
            # Handle literal block marker ::
            if line.rstrip().endswith('::'):
                # Add the line without ::
                text = line.rstrip()[:-2].strip()
                if text:
                    output_lines.append(text)
                output_lines.append('')
                output_lines.append('[CODE]')
                in_code_block = True
                directive_indent = 3
                i += 1
                # Skip blank lines after ::
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    directive_indent = len(lines[i]) - len(lines[i].lstrip())
                continue
            
            # Handle other directives (skip most)
            if re.match(r'^\.\.\s+\w+::', line):
                in_directive = True
                directive_indent = len(line) - len(line.lstrip()) + 3
                i += 1
                continue
            
            # End code block/directive when we return to normal indentation
            if (in_code_block or in_directive) and line.strip():
                current_indent = len(line) - len(line.lstrip())
                if current_indent < directive_indent:
                    if in_code_block:
                        output_lines.append('[/CODE]')
                        output_lines.append('')
                    in_code_block = False
                    in_directive = False
            
            if in_code_block:
                # Remove common indent from code
                if line.strip():
                    code_line = line[directive_indent:] if len(line) > directive_indent else line
                    output_lines.append(code_line)
                else:
                    output_lines.append('')
                i += 1
                continue
            
            if in_directive:
                i += 1
                continue
            
            # Handle RST headings (underlined titles)
            if i + 1 < len(lines) and line.strip():
                next_line = lines[i + 1]
                underline_char = None
                
                for char in ['=', '-', '~', '^', '"', '*', '+', '#']:
                    if re.match(f'^{re.escape(char)}+\\s*$', next_line):
                        if len(next_line.strip()) >= len(line.strip()):
                            underline_char = char
                            break
                
                if underline_char:
                    output_lines.append('')
                    output_lines.append(line.strip())
                    if underline_char in '=#':
                        output_lines.append('=' * min(60, len(line.strip())))
                    else:
                        output_lines.append('-' * min(50, len(line.strip())))
                    output_lines.append('')
                    i += 2
                    continue
            
            # Process inline formatting
            processed = self._process_rst_inline(line)
            output_lines.append(processed)
            i += 1
        
        # Close any unclosed code block
        if in_code_block:
            output_lines.append('[/CODE]')
        
        return self._clean_output('\n'.join(output_lines))
    
    def _process_rst_inline(self, line: str) -> str:
        """Process inline RST formatting."""
        # Remove inline literals (keep content)
        line = re.sub(r'``(.+?)``', r'`\1`', line)
        
        # Remove emphasis
        line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
        line = re.sub(r'\*(.+?)\*', r'\1', line)
        
        # Remove interpreted text roles
        line = re.sub(r':\w+:`([^`]+)`', r'\1', line)
        
        # Remove reference markers
        line = re.sub(r'`([^`]+)`_+', r'\1', line)
        line = re.sub(r'`([^`<]+)\s*<[^>]+>`_+', r'\1', line)
        
        # Remove footnote/citation references
        line = re.sub(r'\[#?\w*\]_', '', line)
        
        return line
    
    def _clean_output(self, text: str) -> str:
        """Clean up the converted output."""
        # Remove excessive blank lines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Remove trailing whitespace
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove leading/trailing blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)


def convert_md_rst(path: str) -> Tuple[Optional[str], str]:
    """Convenience function to convert a Markdown or RST file."""
    converter = MarkdownRSTConverter()
    return converter.convert(path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python md_rst_to_txt.py <markdown_or_rst_file>")
        sys.exit(1)
    
    converter = MarkdownRSTConverter()
    
    for path in sys.argv[1:]:
        content, status = converter.convert(path)
        if content:
            print(f"=== {path} ({status}) ===")
            print(content[:3000])
            print(f"... ({len(content)} chars total)")
        else:
            print(f"Failed to convert {path}: {status}")

