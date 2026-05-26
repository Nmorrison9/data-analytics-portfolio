"""
Loan Default Risk Analysis
==========================
Analyzes loan portfolio data by visualizing credit score distributions,
comparing default rates across credit score and interest rate buckets,
and identifying feature correlations using a Pearson-based heatmap.

Author : Nathaniel Morrison
Date   : 05/20/2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH = "data/loan_data.csv"

CREDIT_SCORE_BINS  = [300, 579, 669, 739, 799, 850]
CREDIT_SCORE_LABELS = ["Poor\n(300–579)", "Fair\n(580–669)",
                        "Good\n(670–739)", "Very Good\n(740–799)",
                        "Exceptional\n(800–850)"]

INTEREST_RATE_BINS = 5          # equal-width buckets
PCT_FORMATTER      = mticker.PercentFormatter(xmax=1, decimals=0)

# Shared visual style
PALETTE   = "#2563EB"           # bar fill
BAR_EDGE  = "#1E3A8A"           # bar edge
GRID_CLR  = "#E5E7EB"           # subtle horizontal gridlines
FIG_SIZE  = (8, 5)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load the loan CSV and print a basic summary to stdout."""
    df = pd.read_csv(path)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    df.info(show_counts=True)

    print("\nSummary statistics:")
    print(df.describe())

    default_counts = df["default"].value_counts()
    default_rate   = df["default"].mean()
    print(f"\nDefault counts:\n{default_counts}")
    print(f"\nOverall default rate: {default_rate:.2%}")

    return df


def style_bar_ax(ax: plt.Axes, title: str, xlabel: str,
                 ylabel: str = "Default Rate") -> None:
    """Apply consistent styling to a bar-chart Axes object."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(PCT_FORMATTER)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=0)


def add_value_labels(ax: plt.Axes) -> None:
    """Annotate each bar with its percentage value."""
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.003,
            f"{height:.1%}",
            ha="center", va="bottom", fontsize=9, color="#374151",
        )


# ── Analysis functions ────────────────────────────────────────────────────────

def plot_credit_score_distribution(df: pd.DataFrame) -> None:
    """Plot a histogram of credit score values."""
    scores = df["credit_score"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.hist(scores, bins="auto", color=PALETTE, edgecolor=BAR_EDGE, linewidth=0.6)

    ax.set_title("Credit Score Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Credit Score", fontsize=11)
    ax.set_ylabel("Number of Borrowers", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


def plot_default_by_credit_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin borrowers into standard credit-score tiers and plot the mean
    default rate per tier.

    Returns the DataFrame with a new 'credit_bucket' column.
    """
    df["credit_bucket"] = pd.cut(
        df["credit_score"],
        bins=CREDIT_SCORE_BINS,
        labels=CREDIT_SCORE_LABELS,
    )

    default_rate = df.groupby("credit_bucket", observed=True)["default"].mean()

    print("\nDefault rate by credit-score bucket:")
    print(default_rate.to_string())

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(
        default_rate.index.astype(str),
        default_rate.values,
        color=PALETTE,
        edgecolor=BAR_EDGE,
        linewidth=0.6,
        width=0.6,
    )

    style_bar_ax(ax, "Default Rate by Credit Score Tier", "Credit Score Tier")
    add_value_labels(ax)
    fig.tight_layout()
    plt.show()

    return df


def plot_default_by_interest_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin loans into five equal-width interest-rate brackets and plot
    the mean default rate per bracket.

    Returns the DataFrame with a new 'interest_rate_bucket' column.
    """
    df["interest_rate_bucket"] = pd.cut(df["interest_rate"], bins=INTEREST_RATE_BINS)

    default_rate = df.groupby("interest_rate_bucket", observed=True)["default"].mean()
    bucket_counts = df.groupby("interest_rate_bucket", observed=True)["default"].count()

    print("\nDefault rate by interest-rate bucket:")
    print(default_rate.to_string())
    print("\nLoan count per interest-rate bucket:")
    print(bucket_counts.to_string())

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(
        default_rate.index.astype(str),
        default_rate.values,
        color=PALETTE,
        edgecolor=BAR_EDGE,
        linewidth=0.6,
        width=0.6,
    )

    style_bar_ax(ax, "Default Rate by Interest Rate", "Interest Rate Range")
    add_value_labels(ax)
    ax.tick_params(axis="x", rotation=-30)
    fig.tight_layout()
    plt.show()

    return df


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Compute a Pearson correlation matrix on all numeric features
    (excluding helper bucket columns) and display it as a heatmap.
    """
    numeric_df = (
        df.select_dtypes(exclude=["object", "category"])
          .drop(columns=["credit_bucket", "interest_rate_bucket"], errors="ignore")
    )

    corr_matrix = numeric_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="#E5E7EB",
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix (Pearson)", fontsize=13,
                 fontweight="bold", pad=14)
    fig.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    df = load_data(DATA_PATH)

    plot_credit_score_distribution(df)
    df = plot_default_by_credit_bucket(df)
    df = plot_default_by_interest_bucket(df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()
