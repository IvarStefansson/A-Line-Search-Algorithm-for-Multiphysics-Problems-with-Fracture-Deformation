import os
from collections import namedtuple
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def heatmap(
    df: pd.DataFrame,
    a: int,
    b: int,
    file_name: str,
    title: str = None,
    figsize: Tuple[int, int] = (8, 4),
    x_label: str = None,
):
    """Create a heatmap of the number of iterations.

    Parameters:
        df: DataFrame with the number of iterations.
        a: Lower limit for the color map.
        b: Upper limit for the color map.
        file_name: Name of the file to save the plot to.
        title: Title of the plot.
        figsize: Size of the figure.
        x_label: Label for the x-axis.

    """
    # Convert to integer
    df = df.astype(int)
    # Create a custom color map. Two first colors for -1 and -2
    cmap_it = sns.diverging_palette(150, 10, as_cmap=True)
    cmap_it.set_over("grey")
    cmap_it.set_under("darkgrey")
    # Create annotation with the number of iterations.
    # Replace -1, 200 and 250 with "Not converged", "Diverged" and "stuck"
    annotation = (
        df.astype(str)
        # .replace("-1", "MaxIt")
        .replace("-200", "Div")
        .replace("250", "Stuck")
        .replace(f"{b+1}", "NC")
    )
    # Create a heatmap with grid lines
    plt.figure(figsize=figsize)
    cax = sns.heatmap(
        df,
        annot=annotation,
        cmap=cmap_it,
        cbar=True,
        vmin=a,
        vmax=b,
        fmt="",
        linewidths=1.0,
        linecolor="black",
        annot_kws={"size": 12, "weight": "bold", "color": "black"},
    )

    # Add a color bar legend.
    cbar = cax.collections[0].colorbar
    cbar.set_label("Number of iterations", fontsize=12)
    ticks = np.linspace(a, b, 2)
    cbar.set_ticks(ticks)
    labels = [str(int(i)) for i in ticks]
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=12)
    cbar.set_ticklabels(labels)
    # Flip the axis0 labels
    plt.gca().tick_params(axis="y", rotation=0)
    # Add a title
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    folder_name = file_name.split("/")[:-1]
    folder_name = "/".join(folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(file_name, dpi=300, bbox_inches="tight")


def extract_table_data(data: dict, ax_names: namedtuple) -> dict:
    # Loop over all combinations of axes values and extract the number of iterations.
    all_data = {}
    for data_point in data:
        key = []
        for ax in ax_names:
            key.append(data_point[ax])
        key = tuple(key)
        all_data[key] = data_point["iterations"]
    return all_data


def strip_leading_zero_scientific(value: float, value_digits=1) -> str:
    """Format a value in scientific notation without leading zeros.

    Parameters:
        value: The value to format.
        value_digits: The number of digits to include in the value.

    Returns:
        The formatted value in scientific notation without leading zeros.

    """
    # Format the value in scientific notation
    formatted_value = f"{value:.{value_digits}e}"
    # Split the mantissa and exponent
    mantissa, exponent = formatted_value.split("e")
    # Track the sign of the exponent
    sign = exponent[0]
    # Remove the leading zero from the exponent
    exponent = exponent[2:]
    # Reconstruct the value in scientific notation
    return f"{mantissa}e{sign}{exponent}"
