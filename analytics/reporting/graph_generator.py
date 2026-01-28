"""
Professional Graph Generation System

Provides high-quality, professional graph generation using matplotlib
with customizable styling and comprehensive chart type support.

Features:
- Line charts (time-series data)
- Bar charts (comparisons)
- Pie charts (distributions)
- Scatter plots (correlations)
- Stacked bar charts (compositional data)
- Professional styling and formatting
- High-resolution output (300 DPI)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from .styles import (
    apply_professional_style, get_color_palette, get_figure_size,
    colors, fonts, graph_styles, layouts
)


class GraphGenerator:
    """
    Professional graph generation system for analytics data visualization.

    Creates high-quality, publication-ready charts with consistent styling
    and professional appearance suitable for technical documentation.
    """

    def __init__(self, output_dir: str = "graphs", dpi: int = 300):
        """
        Initialize graph generator.

        Args:
            output_dir: Directory to save generated graphs
            dpi: Output resolution (dots per inch)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dpi = dpi

        # Apply professional styling
        apply_professional_style()

        # Graph registry
        self.generated_graphs = {}

    def create_line_chart(
        self,
        data: Dict[str, List[Tuple[datetime, float]]],
        title: str,
        xlabel: str = "Time",
        ylabel: str = "Value",
        filename: Optional[str] = None,
        size: str = "medium",
        show_legend: bool = True,
        show_grid: bool = True,
        trend_line: bool = False
    ) -> str:
        """
        Create a professional line chart (time-series).

        Args:
            data: Dictionary mapping series names to list of (timestamp, value) tuples
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Output filename (auto-generated if None)
            size: Chart size ('small', 'medium', 'large', 'wide', 'tall')
            show_legend: Whether to show legend
            show_grid: Whether to show grid
            trend_line: Whether to add trend lines

        Returns:
            Path to saved graph file
        """
        fig, ax = plt.subplots(figsize=get_figure_size(size))

        # Get color palette
        color_palette = get_color_palette(len(data))

        # Plot each series
        for i, (series_name, series_data) in enumerate(data.items()):
            if not series_data:
                continue

            timestamps, values = zip(*series_data)

            # Create line plot
            line_style = graph_styles['line_chart']
            ax.plot(
                timestamps, values,
                color=color_palette[i],
                linewidth=line_style['linewidth'],
                marker=line_style['marker_style'],
                markersize=line_style['markersize'],
                alpha=line_style['alpha'],
                label=series_name
            )

            # Add trend line if requested
            if trend_line and len(values) > 1:
                # Calculate trend line
                x_numeric = mdates.date2num(timestamps)
                z = np.polyfit(x_numeric, values, 1)
                p = np.poly1d(z)
                ax.plot(
                    timestamps, p(x_numeric),
                    color=color_palette[i],
                    linestyle='--',
                    alpha=0.5,
                    linewidth=1
                )

        # Format x-axis for datetime
        if any(series_data for series_data in data.values()):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Set labels and title
        ax.set_title(title, fontdict=fonts['title'], pad=20)
        ax.set_xlabel(xlabel, fontdict=fonts['axis_label'])
        ax.set_ylabel(ylabel, fontdict=fonts['axis_label'])

        # Add legend
        if show_legend and len(data) > 1:
            ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)

        # Add grid
        if show_grid:
            ax.grid(True, alpha=layouts['grid']['alpha'],
                   linestyle=layouts['grid']['linestyle'])

        # Adjust layout
        plt.tight_layout()

        # Save graph
        if filename is None:
            filename = f"line_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor=colors['background'])
        plt.close()

        self.generated_graphs[filename] = str(filepath)
        return str(filepath)

    def create_bar_chart(
        self,
        data: Dict[str, List[float]],
        title: str,
        xlabel: str = "Categories",
        ylabel: str = "Values",
        filename: Optional[str] = None,
        size: str = "medium",
        orientation: str = "vertical",
        show_values: bool = True,
        show_grid: bool = True
    ) -> str:
        """
        Create a professional bar chart.

        Args:
            data: Dictionary mapping series names to lists of values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Output filename (auto-generated if None)
            size: Chart size
            orientation: 'vertical' or 'horizontal'
            show_values: Whether to show value labels on bars
            show_grid: Whether to show grid

        Returns:
            Path to saved graph file
        """
        fig, ax = plt.subplots(figsize=get_figure_size(size))

        # Get data dimensions
        categories = list(data.keys())
        values = list(data.values())
        color_palette = get_color_palette(len(categories))

        bar_style = graph_styles['bar_chart']

        if orientation == 'vertical':
            bars = ax.bar(
                categories, values,
                color=color_palette,
                width=bar_style['bar_width'],
                alpha=bar_style['alpha'],
                edgecolor=bar_style['edge_color'],
                linewidth=bar_style['edge_width']
            )

            # Add value labels
            if show_values:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                           f'{value:.1f}', ha='center', va='bottom',
                           fontdict=fonts['annotation'])

            ax.set_ylabel(ylabel, fontdict=fonts['axis_label'])
            ax.set_xlabel(xlabel, fontdict=fonts['axis_label'])
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        else:  # horizontal
            bars = ax.barh(
                categories, values,
                color=color_palette,
                height=bar_style['bar_width'],
                alpha=bar_style['alpha'],
                edgecolor=bar_style['edge_color'],
                linewidth=bar_style['edge_width']
            )

            # Add value labels
            if show_values:
                for bar, value in zip(bars, values):
                    width = bar.get_width()
                    ax.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2.,
                           f'{value:.1f}', ha='left', va='center',
                           fontdict=fonts['annotation'])

            ax.set_xlabel(ylabel, fontdict=fonts['axis_label'])
            ax.set_ylabel(xlabel, fontdict=fonts['axis_label'])

        # Set title
        ax.set_title(title, fontdict=fonts['title'], pad=20)

        # Add grid
        if show_grid:
            if orientation == 'vertical':
                ax.yaxis.grid(True, alpha=layouts['grid']['alpha'],
                            linestyle=layouts['grid']['linestyle'])
            else:
                ax.xaxis.grid(True, alpha=layouts['grid']['alpha'],
                            linestyle=layouts['grid']['linestyle'])

        # Adjust layout
        plt.tight_layout()

        # Save graph
        if filename is None:
            filename = f"bar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor=colors['background'])
        plt.close()

        self.generated_graphs[filename] = str(filepath)
        return str(filepath)

    def create_pie_chart(
        self,
        data: Dict[str, float],
        title: str,
        filename: Optional[str] = None,
        size: str = "medium",
        show_percentages: bool = True,
        show_legend: bool = True,
        explode_largest: bool = False
    ) -> str:
        """
        Create a professional pie chart.

        Args:
            data: Dictionary mapping labels to values
            title: Chart title
            filename: Output filename (auto-generated if None)
            size: Chart size
            show_percentages: Whether to show percentage labels
            show_legend: Whether to show legend
            explode_largest: Whether to explode the largest segment

        Returns:
            Path to saved graph file
        """
        fig, ax = plt.subplots(figsize=get_figure_size(size))

        # Prepare data
        labels = list(data.keys())
        values = list(data.values())
        color_palette = get_color_palette(len(labels))

        # Configure explosion
        explode = None
        if explode_largest and values:
            max_idx = values.index(max(values))
            explode = [0.1 if i == max_idx else 0 for i in range(len(values))]
        else:
            explode = [0] * len(values)

        pie_style = graph_styles['pie_chart']

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values, labels=labels if not show_legend else None,
            colors=color_palette, explode=explode,
            autopct='%1.1f%%' if show_percentages else None,
            shadow=pie_style['shadow'],
            startangle=pie_style['start_angle'],
            textprops=pie_style['text_props']
        )

        # Set title
        ax.set_title(title, fontdict=fonts['title'], pad=20)

        # Add legend
        if show_legend:
            ax.legend(wedges, labels,
                     loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                     framealpha=0.9, fancybox=True, shadow=True)

        # Ensure aspect ratio is equal
        ax.axis('equal')

        # Save graph
        if filename is None:
            filename = f"pie_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor=colors['background'])
        plt.close()

        self.generated_graphs[filename] = str(filepath)
        return str(filepath)

    def create_stacked_bar_chart(
        self,
        data: Dict[str, Dict[str, float]],
        title: str,
        xlabel: str = "Categories",
        ylabel: str = "Values",
        filename: Optional[str] = None,
        size: str = "medium",
        show_legend: bool = True,
        show_grid: bool = True
    ) -> str:
        """
        Create a professional stacked bar chart.

        Args:
            data: Nested dictionary {category: {series: value}}
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Output filename (auto-generated if None)
            size: Chart size
            show_legend: Whether to show legend
            show_grid: Whether to show grid

        Returns:
            Path to saved graph file
        """
        fig, ax = plt.subplots(figsize=get_figure_size(size))

        # Prepare data
        categories = list(data.keys())
        series_names = list(next(iter(data.values())).keys())
        color_palette = get_color_palette(len(series_names))

        # Create stacked bars
        bottom_values = [0] * len(categories)
        bar_style = graph_styles['bar_chart']

        for i, series in enumerate(series_names):
            values = [data[cat].get(series, 0) for cat in categories]

            ax.bar(
                categories, values,
                bottom=bottom_values,
                color=color_palette[i],
                alpha=bar_style['alpha'],
                edgecolor=bar_style['edge_color'],
                linewidth=bar_style['edge_width'],
                label=series
            )

            # Update bottom values for next stack
            bottom_values = [bottom + val for bottom, val in zip(bottom_values, values)]

        # Set labels and title
        ax.set_title(title, fontdict=fonts['title'], pad=20)
        ax.set_xlabel(xlabel, fontdict=fonts['axis_label'])
        ax.set_ylabel(ylabel, fontdict=fonts['axis_label'])

        # Add legend
        if show_legend:
            ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)

        # Add grid
        if show_grid:
            ax.yaxis.grid(True, alpha=layouts['grid']['alpha'],
                        linestyle=layouts['grid']['linestyle'])

        # Rotate x-axis labels if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Adjust layout
        plt.tight_layout()

        # Save graph
        if filename is None:
            filename = f"stacked_bar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor=colors['background'])
        plt.close()

        self.generated_graphs[filename] = str(filepath)
        return str(filepath)

    def create_system_resource_dashboard(
        self,
        cpu_data: List[Tuple[datetime, float]],
        memory_data: List[Tuple[datetime, float]],
        disk_data: Optional[List[Tuple[datetime, float]]] = None,
        title: str = "System Resource Usage",
        filename: Optional[str] = None,
        size: str = "large"
    ) -> str:
        """
        Create a comprehensive system resource dashboard.

        Args:
            cpu_data: List of (timestamp, cpu_percent) tuples
            memory_data: List of (timestamp, memory_mb) tuples
            disk_data: Optional list of (timestamp, disk_io_mb) tuples
            title: Dashboard title
            filename: Output filename (auto-generated if None)
            size: Chart size

        Returns:
            Path to saved graph file
        """
        fig, axes = plt.subplots(2, 2, figsize=get_figure_size(size))
        fig.suptitle(title, fontdict=fonts['title'], fontsize=18)

        # CPU Usage
        if cpu_data:
            timestamps, cpu_values = zip(*cpu_data)
            axes[0, 0].plot(timestamps, cpu_values, color=colors['primary'],
                           linewidth=2.5, marker='o', markersize=4)
            axes[0, 0].set_title('CPU Usage', fontdict=fonts['subtitle'])
            axes[0, 0].set_ylabel('CPU %', fontdict=fonts['axis_label'])
            axes[0, 0].grid(True, alpha=layouts['grid']['alpha'])
            axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

        # Memory Usage
        if memory_data:
            timestamps, memory_values = zip(*memory_data)
            axes[0, 1].plot(timestamps, memory_values, color=colors['secondary'],
                           linewidth=2.5, marker='s', markersize=4)
            axes[0, 1].set_title('Memory Usage', fontdict=fonts['subtitle'])
            axes[0, 1].set_ylabel('Memory (MB)', fontdict=fonts['axis_label'])
            axes[0, 1].grid(True, alpha=layouts['grid']['alpha'])
            axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

        # Disk I/O (if available)
        if disk_data:
            timestamps, disk_values = zip(*disk_data)
            axes[1, 0].plot(timestamps, disk_values, color=colors['tertiary'],
                           linewidth=2.5, marker='^', markersize=4)
            axes[1, 0].set_title('Disk I/O', fontdict=fonts['subtitle'])
            axes[1, 0].set_ylabel('Disk I/O (MB)', fontdict=fonts['axis_label'])
            axes[1, 0].grid(True, alpha=layouts['grid']['alpha'])
            axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Disk Data Available',
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontdict=fonts['annotation'])
            axes[1, 0].set_title('Disk I/O', fontdict=fonts['subtitle'])

        # Summary Statistics
        axes[1, 1].axis('off')
        summary_text = "Resource Summary\n" + "="*20 + "\n"

        if cpu_data:
            cpu_values = [v for _, v in cpu_data]
            summary_text += f"CPU: Avg {np.mean(cpu_values):.1f}%, "
            summary_text += f"Peak {np.max(cpu_values):.1f}%\n"

        if memory_data:
            memory_values = [v for _, v in memory_data]
            summary_text += f"Memory: Avg {np.mean(memory_values):.1f} MB, "
            summary_text += f"Peak {np.max(memory_values):.1f} MB\n"

        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontdict=fonts['axis_label'], verticalalignment='top')

        plt.tight_layout()

        # Save graph
        if filename is None:
            filename = f"system_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor=colors['background'])
        plt.close()

        self.generated_graphs[filename] = str(filepath)
        return str(filepath)

    def get_generated_graphs(self) -> Dict[str, str]:
        """Get dictionary of all generated graphs"""
        return self.generated_graphs.copy()

    def clear_graphs(self):
        """Clear the graph registry"""
        self.generated_graphs.clear()