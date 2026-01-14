#!/usr/bin/env python
"""
Convert Texinfo (.texi, .texinfo) files to plaintext.

Texinfo is GNU's documentation format used by many GNU projects
including Bash, GCC, Emacs, etc.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, List


class TexinfoConverter:
    """Convert Texinfo documents to clean plaintext."""
    
    TEXINFO_EXTENSIONS = {'.texi', '.texinfo', '.txi'}
    
    def __init__(self):
        pass
    
    def is_texinfo(self, path: str) -> bool:
        """Check if file is a Texinfo file."""
        return Path(path).suffix.lower() in self.TEXINFO_EXTENSIONS
    
    def convert(self, path: str) -> Tuple[Optional[str], str]:
        """
        Convert a Texinfo file to plaintext.
        
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
            
            result = self._convert_texinfo(content)
            return result, "texinfo_parser"
            
        except Exception as e:
            return None, f"Error: {e}"
    
    def _convert_texinfo(self, content: str) -> str:
        """Convert Texinfo markup to plaintext."""
        lines = content.split('\n')
        output_lines: List[str] = []
        
        in_example = False
        in_verbatim = False
        skip_until_end = None
        
        for line in lines:
            stripped = line.strip()
            
            # Handle block environments we want to skip
            if skip_until_end:
                if stripped == skip_until_end:
                    skip_until_end = None
                continue
            
            # Skip certain blocks entirely
            if stripped.startswith('@ignore'):
                skip_until_end = '@end ignore'
                continue
            if stripped.startswith('@ifnotinfo') or stripped.startswith('@ifhtml'):
                skip_until_end = '@end ' + stripped[1:].split()[0]
                continue
            
            # Handle example/verbatim blocks (preserve content)
            if stripped.startswith('@example') or stripped.startswith('@verbatim'):
                in_example = True
                output_lines.append('')
                output_lines.append('[CODE]')
                continue
            if stripped.startswith('@end example') or stripped.startswith('@end verbatim'):
                in_example = False
                output_lines.append('[/CODE]')
                output_lines.append('')
                continue
            
            if in_example:
                # Preserve code content, just remove @ escapes
                code_line = self._process_inline_commands(line)
                output_lines.append(code_line)
                continue
            
            # Skip Texinfo directives/commands we don't render
            if self._should_skip_line(stripped):
                continue
            
            # Handle section headers
            header = self._extract_header(stripped)
            if header:
                output_lines.append('')
                output_lines.append(header)
                output_lines.append('')
                continue
            
            # Handle @item in lists
            if stripped.startswith('@item'):
                item_text = stripped[5:].strip()
                item_text = self._process_inline_commands(item_text)
                output_lines.append(f"  - {item_text}" if item_text else "  -")
                continue
            
            # Regular text - process inline commands
            processed = self._process_inline_commands(line)
            output_lines.append(processed)
        
        # Clean up
        result = '\n'.join(output_lines)
        result = self._clean_output(result)
        
        return result
    
    def _should_skip_line(self, line: str) -> bool:
        """Check if line is a Texinfo directive we should skip."""
        skip_commands = [
            '@c ', '@comment',  # Comments
            '@setfilename', '@settitle', '@set ',
            '@setchapternewpage', '@setcontentsafternewpage',
            '@paragraphindent', '@firstparagraphindent',
            '@dircategory', '@direntry',
            '@copying', '@end copying',
            '@titlepage', '@end titlepage',
            '@contents', '@summarycontents', '@shortcontents',
            '@node', '@menu', '@end menu',
            '@detailmenu', '@end detailmenu',
            '@ifinfo', '@end ifinfo',
            '@ifnottex', '@end ifnottex',
            '@tex', '@end tex',
            '@html', '@end html',
            '@documentencoding', '@documentlanguage',
            '@finalout', '@bye',
            '@vskip', '@page', '@sp ',
            '@need', '@group', '@end group',
            '@include',
            '@printindex', '@syncodeindex', '@synindex',
            '@defindex', '@defcodeindex',
            '@cindex', '@findex', '@vindex', '@pindex', '@tindex', '@kindex',
            '@anchor',
        ]
        
        for cmd in skip_commands:
            if line.startswith(cmd):
                return True
        
        # Skip lines that are just @end something (for blocks we process)
        if line.startswith('@end '):
            return True
        
        # Skip @-only lines or common block starts we handle elsewhere
        if line in ('@', '@itemize', '@enumerate', '@table', '@ftable', '@vtable',
                    '@multitable', '@end itemize', '@end enumerate', '@end table',
                    '@end ftable', '@end vtable', '@end multitable'):
            return True
        
        return False
    
    def _extract_header(self, line: str) -> Optional[str]:
        """Extract section header from Texinfo command."""
        header_commands = {
            '@chapter': '# ',
            '@unnumbered': '# ',
            '@appendix': '# Appendix: ',
            '@section': '## ',
            '@unnumberedsec': '## ',
            '@appendixsec': '## ',
            '@subsection': '### ',
            '@unnumberedsubsec': '### ',
            '@appendixsubsec': '### ',
            '@subsubsection': '#### ',
            '@unnumberedsubsubsec': '#### ',
            '@appendixsubsubsec': '#### ',
            '@heading': '## ',
            '@subheading': '### ',
            '@subsubheading': '#### ',
            '@majorheading': '# ',
            '@chapheading': '# ',
            '@top': '# ',
        }
        
        for cmd, prefix in header_commands.items():
            if line.startswith(cmd + ' ') or line.startswith(cmd + '\t'):
                title = line[len(cmd):].strip()
                title = self._process_inline_commands(title)
                return prefix + title
            elif line == cmd:
                return prefix + "(Untitled)"
        
        return None
    
    def _process_inline_commands(self, text: str) -> str:
        """Process inline Texinfo commands."""
        # @code{text} -> `text`
        text = re.sub(r'@code\{([^}]*)\}', r'`\1`', text)
        text = re.sub(r'@samp\{([^}]*)\}', r'`\1`', text)
        text = re.sub(r'@verb\{([^}]*)\}', r'`\1`', text)
        text = re.sub(r'@command\{([^}]*)\}', r'`\1`', text)
        text = re.sub(r'@option\{([^}]*)\}', r'`\1`', text)
        text = re.sub(r'@env\{([^}]*)\}', r'`\1`', text)
        text = re.sub(r'@file\{([^}]*)\}', r'`\1`', text)
        text = re.sub(r'@kbd\{([^}]*)\}', r'`\1`', text)
        text = re.sub(r'@key\{([^}]*)\}', r'[\1]', text)
        
        # @var{text} -> <text> (variable placeholder)
        text = re.sub(r'@var\{([^}]*)\}', r'<\1>', text)
        
        # @emph{text}, @strong{text} -> text (remove emphasis markers)
        text = re.sub(r'@emph\{([^}]*)\}', r'\1', text)
        text = re.sub(r'@strong\{([^}]*)\}', r'\1', text)
        text = re.sub(r'@i\{([^}]*)\}', r'\1', text)
        text = re.sub(r'@b\{([^}]*)\}', r'\1', text)
        text = re.sub(r'@r\{([^}]*)\}', r'\1', text)
        text = re.sub(r'@t\{([^}]*)\}', r'\1', text)
        text = re.sub(r'@sc\{([^}]*)\}', r'\1', text)
        text = re.sub(r'@dfn\{([^}]*)\}', r'\1', text)
        text = re.sub(r'@cite\{([^}]*)\}', r'"\1"', text)
        
        # @url{url} or @url{url, text}
        text = re.sub(r'@url\{([^,}]+)(?:,[^}]*)?\}', r'\1', text)
        text = re.sub(r'@uref\{([^,}]+)(?:,[^}]*)?\}', r'\1', text)
        text = re.sub(r'@email\{([^,}]+)(?:,[^}]*)?\}', r'\1', text)
        
        # @xref, @pxref, @ref -> See: text
        text = re.sub(r'@[px]?ref\{([^,}]+)(?:,[^}]*)?\}', r'(see \1)', text)
        
        # @footnote{text} -> (Note: text)
        text = re.sub(r'@footnote\{([^}]*)\}', r'(Note: \1)', text)
        
        # @w{text} -> text (prevent line break - just remove)
        text = re.sub(r'@w\{([^}]*)\}', r'\1', text)
        
        # @: -> (nothing - prevent sentence-end space)
        text = text.replace('@:', '')
        
        # @@ -> @
        text = text.replace('@@', '@')
        
        # @{ -> {, @} -> }
        text = text.replace('@{', '{')
        text = text.replace('@}', '}')
        
        # @. @? @! -> . ? !
        text = text.replace('@.', '.')
        text = text.replace('@?', '?')
        text = text.replace('@!', '!')
        
        # @* -> (line break, but we'll handle with newline)
        text = text.replace('@*', '\n')
        
        # @minus{} -> -
        text = re.sub(r'@minus\{\}', '-', text)
        
        # @bullet{} -> •
        text = re.sub(r'@bullet\{\}', '•', text)
        
        # @dots{} -> ...
        text = re.sub(r'@dots\{\}', '...', text)
        
        # @result{} -> =>
        text = re.sub(r'@result\{\}', '=>', text)
        
        # @expansion{} -> ==>
        text = re.sub(r'@expansion\{\}', '==>', text)
        
        # @print{} -> -|
        text = re.sub(r'@print\{\}', '-|', text)
        
        # @error{} -> error-->
        text = re.sub(r'@error\{\}', 'error-->', text)
        
        # Remove any remaining @command{} patterns
        text = re.sub(r'@\w+\{([^}]*)\}', r'\1', text)
        
        return text
    
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


def convert_texinfo(path: str) -> Tuple[Optional[str], str]:
    """Convenience function to convert a single Texinfo file."""
    converter = TexinfoConverter()
    return converter.convert(path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python texinfo_to_txt.py <texinfo_file>")
        sys.exit(1)
    
    converter = TexinfoConverter()
    
    for path in sys.argv[1:]:
        content, status = converter.convert(path)
        if content:
            print(f"=== {path} ({status}) ===")
            print(content[:3000])
            print(f"... ({len(content)} chars total)")
        else:
            print(f"Failed to convert {path}: {status}")

