#!/usr/bin/env python
"""
Convert Unix man pages to plaintext.

Supports:
- .1, .2, .3, .5, .7, .8 sections (and .gz compressed variants)
- Uses groff + col for conversion when available
- Falls back to basic roff parsing if groff unavailable
"""

import gzip
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


class ManPageConverter:
    """Convert man pages to clean plaintext."""
    
    # Man page section extensions
    MAN_EXTENSIONS = {'.1', '.2', '.3', '.5', '.7', '.8', '.1p', '.3p'}
    
    def __init__(self):
        self.groff_available = self._check_groff()
        self.col_available = self._check_col()
        
    def _check_groff(self) -> bool:
        """Check if groff is available."""
        return shutil.which('groff') is not None
    
    def _check_col(self) -> bool:
        """Check if col is available."""
        return shutil.which('col') is not None
    
    def is_manpage(self, path: str) -> bool:
        """Check if a file is a man page based on extension."""
        p = Path(path)
        
        # Handle .gz compression
        if p.suffix == '.gz':
            p = Path(p.stem)
        
        return p.suffix.lower() in self.MAN_EXTENSIONS
    
    def convert(self, path: str) -> Tuple[Optional[str], str]:
        """
        Convert a man page to plaintext.
        
        Returns:
            Tuple of (content, status_message)
            content is None if conversion failed
        """
        path = Path(path)
        
        if not path.exists():
            return None, f"File not found: {path}"
        
        try:
            # Read content (handle gzip)
            if path.suffix == '.gz':
                with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
                    raw_content = f.read()
            else:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    raw_content = f.read()
            
            # Try groff conversion first
            if self.groff_available:
                result = self._convert_with_groff(raw_content)
                if result:
                    return self._clean_manpage_text(result), "groff"
            
            # Fallback to basic parsing
            result = self._parse_roff_basic(raw_content)
            return self._clean_manpage_text(result), "basic_parser"
            
        except Exception as e:
            return None, f"Error: {e}"
    
    def _convert_with_groff(self, content: str) -> Optional[str]:
        """Convert using groff -Tutf8 -man | col -bx."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.man', delete=False, 
                                             encoding='utf-8') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                # groff -Tutf8 -man <file>
                groff_cmd = ['groff', '-Tutf8', '-man', tmp_path]
                groff_proc = subprocess.Popen(
                    groff_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                
                if self.col_available:
                    # Pipe through col -bx to remove backspaces
                    col_proc = subprocess.Popen(
                        ['col', '-bx'],
                        stdin=groff_proc.stdout,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL
                    )
                    groff_proc.stdout.close()
                    output, _ = col_proc.communicate(timeout=30)
                else:
                    output, _ = groff_proc.communicate(timeout=30)
                
                return output.decode('utf-8', errors='replace')
                
            finally:
                os.unlink(tmp_path)
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return None
    
    def _parse_roff_basic(self, content: str) -> str:
        """
        Basic roff/troff parser for when groff is unavailable.
        Extracts readable text from man page source.
        """
        lines = []
        in_code_block = False
        
        for line in content.split('\n'):
            line = line.rstrip()
            
            # Skip comments
            if line.startswith('.\\"') or line.startswith("'\\\""):
                continue
            
            # Handle common roff macros
            if line.startswith('.'):
                parts = line.split(None, 1)
                macro = parts[0][1:] if parts else ''
                rest = parts[1] if len(parts) > 1 else ''
                
                # Section headers
                if macro in ('SH', 'SS'):
                    header = self._strip_quotes(rest)
                    lines.append('')
                    lines.append(header.upper() if macro == 'SH' else header)
                    lines.append('')
                    
                # Title
                elif macro == 'TH':
                    parts = rest.split()
                    if parts:
                        title = self._strip_quotes(parts[0])
                        lines.append(f"# {title}")
                        lines.append('')
                
                # Paragraphs
                elif macro in ('P', 'PP', 'LP'):
                    lines.append('')
                
                # Line breaks
                elif macro in ('br', 'sp'):
                    lines.append('')
                
                # Indented paragraphs with tags
                elif macro in ('TP', 'IP'):
                    lines.append('')
                
                # Begin/end display (code blocks)
                elif macro in ('nf', 'EX'):
                    in_code_block = True
                    lines.append('')
                elif macro in ('fi', 'EE'):
                    in_code_block = False
                    lines.append('')
                
                # Bold/Italic with arguments
                elif macro in ('B', 'I', 'BI', 'BR', 'IB', 'IR', 'RB', 'RI'):
                    text = self._strip_quotes(rest)
                    if text:
                        lines.append(text)
                
                # Skip other macros but keep any trailing text
                else:
                    continue
            else:
                # Regular text line
                text = self._process_inline_formatting(line)
                if text or in_code_block:
                    lines.append(text)
        
        return '\n'.join(lines)
    
    def _strip_quotes(self, s: str) -> str:
        """Remove surrounding quotes from string."""
        s = s.strip()
        if len(s) >= 2:
            if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
                s = s[1:-1]
        return s
    
    def _process_inline_formatting(self, text: str) -> str:
        """Process inline roff formatting escapes."""
        # Remove common escapes
        text = re.sub(r'\\fB', '', text)  # Bold
        text = re.sub(r'\\fI', '', text)  # Italic
        text = re.sub(r'\\fR', '', text)  # Roman
        text = re.sub(r'\\fP', '', text)  # Previous font
        text = re.sub(r'\\f\d', '', text)  # Numbered fonts
        text = re.sub(r'\\-', '-', text)  # Minus sign
        text = re.sub(r'\\ ', ' ', text)  # Non-breaking space
        text = re.sub(r'\\&', '', text)   # Zero-width character
        text = re.sub(r'\\e', r'\\', text) # Backslash (raw string for replacement)
        text = re.sub(r'\\`', '`', text)  # Grave accent
        text = re.sub(r"\\'", "'", text)  # Apostrophe
        text = re.sub(r'\\[*"]', '', text) # String interpolation
        text = re.sub(r'\\\(..', '', text) # Special characters
        text = re.sub(r'\\\[.+?\]', '', text) # Named glyphs
        
        return text
    
    def _clean_manpage_text(self, text: str) -> str:
        """Clean up converted man page text."""
        # Remove excessive blank lines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove leading/trailing blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)


def convert_manpage(path: str) -> Tuple[Optional[str], str]:
    """Convenience function to convert a single man page."""
    converter = ManPageConverter()
    return converter.convert(path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python man_to_txt.py <manpage_file>")
        sys.exit(1)
    
    converter = ManPageConverter()
    print(f"groff available: {converter.groff_available}")
    print(f"col available: {converter.col_available}")
    print()
    
    for path in sys.argv[1:]:
        content, status = converter.convert(path)
        if content:
            print(f"=== {path} ({status}) ===")
            print(content[:2000])
            print(f"... ({len(content)} chars total)")
        else:
            print(f"Failed to convert {path}: {status}")

