"""
Enhanced Analytics Reporting System

This module provides professional graph generation and markdown reporting
capabilities for the chunking analytics system.

Features:
- Professional graph generation with matplotlib
- Markdown report templates with executive summaries
- Comprehensive visualization support
- Professional styling and formatting
"""

from .graph_generator import GraphGenerator
from .markdown_reporter import MarkdownReporter
from .styles import professional_style, colors, fonts

__all__ = [
    'GraphGenerator',
    'MarkdownReporter',
    'professional_style',
    'colors',
    'fonts'
]