"""
MyPT ASCII Banner - Cyclops Robot Head

Used across all CLI tools and the web application for consistent branding.
"""

# ASCII Art: Cyclops Robot Head
ROBOT_HEAD = """\
o   o
|   |
.-----.
| (o) |
|  -  |
'-----'"""

BANNER_LINE = "=" * 60

def print_banner(title: str = "MyPT", subtitle: str = None, width: int = 60):
    """
    Print the MyPT banner with robot head.
    
    Args:
        title: Main title text (default: "MyPT")
        subtitle: Optional subtitle
        width: Banner width (default: 60)
    """
    print()
    print(BANNER_LINE)
    
    # Print robot head - center each line within the max width
    lines = ROBOT_HEAD.strip().split('\n')
    max_width = max(len(line) for line in lines)
    base_padding = (width - max_width) // 2
    for line in lines:
        # Center shorter lines (antennas) within the robot's width
        line_padding = (max_width - len(line)) // 2
        print(" " * (base_padding + line_padding) + line)
    
    print()
    
    # Print title centered
    title_padding = (width - len(title)) // 2
    print(" " * title_padding + title)
    
    # Print subtitle if provided
    if subtitle:
        sub_padding = (width - len(subtitle)) // 2
        print(" " * sub_padding + subtitle)
    
    print(BANNER_LINE)
    print()


def print_section(title: str, char: str = "-", width: int = 60):
    """Print a section header."""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def get_banner_string(title: str = "MyPT", subtitle: str = None, width: int = 60) -> str:
    """
    Get the banner as a string (useful for logging).
    
    Args:
        title: Main title text
        subtitle: Optional subtitle
        width: Banner width
    
    Returns:
        Banner as string
    """
    lines = []
    lines.append(BANNER_LINE)
    
    robot_lines = ROBOT_HEAD.strip().split('\n')
    max_width = max(len(line) for line in robot_lines)
    base_padding = (width - max_width) // 2
    for line in robot_lines:
        line_padding = (max_width - len(line)) // 2
        lines.append(" " * (base_padding + line_padding) + line)
    
    lines.append("")
    
    title_padding = (width - len(title)) // 2
    lines.append(" " * title_padding + title)
    
    if subtitle:
        sub_padding = (width - len(subtitle)) // 2
        lines.append(" " * sub_padding + subtitle)
    
    lines.append(BANNER_LINE)
    
    return "\n".join(lines)


# Quick access for common banners
def banner_train():
    """Print training banner."""
    print_banner("MyPT Training", "Offline GPT Training Pipeline")


def banner_generate():
    """Print generation banner."""
    print_banner("MyPT Generate", "Text Generation with Trained Models")


def banner_webapp():
    """Print webapp banner."""
    print_banner("MyPT Web Application", "Offline GPT Training & RAG Pipeline")


def banner_workspace():
    """Print workspace/RAG banner."""
    print_banner("MyPT Workspace", "Agentic RAG Assistant")


def banner_dataset():
    """Print dataset preparation banner."""
    print_banner("MyPT Dataset", "Data Preparation Pipeline")


if __name__ == "__main__":
    # Demo all banners
    print_banner("MyPT", "Offline GPT Pipeline for Secure Environments")
    print("\n--- Individual banners ---\n")
    banner_train()
    banner_generate()
    banner_webapp()
    banner_workspace()
    banner_dataset()

