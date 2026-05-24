import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

logger = logging.getLogger(__name__)

_STYLE = "seaborn-v0_8-darkgrid"
_FIG_DPI = 120


class StockPlotter:
    """Produces standard financial visualisations for a stock DataFrame."""

    def __init__(self, df: pd.DataFrame, symbol: str = "", output_dir: str | Path = "data/processed"):
        self.df = df
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            plt.style.use(_STYLE)
        except OSError:
            plt.style.use("ggplot")

    def plot_closing_price(self, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(self.df.index, self.df["Close"], linewidth=1.2, color="steelblue", label="Close")

        for col in [c for c in self.df.columns if c.startswith("MA_")]:
            ax.plot(self.df.index, self.df[col], linewidth=0.9, linestyle="--", label=col)

        ax.set_title(f"{self.symbol} — Closing Price & Moving Averages", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (BRL)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(fontsize=9)
        fig.tight_layout()
        self._save_or_show(fig, "closing_price.png", save)

    def plot_returns_distribution(self, save: bool = True) -> None:
        if "Daily_Return" not in self.df.columns:
            logger.warning("Daily_Return column not found. Skipping returns plot.")
            return

        returns = self.df["Daily_Return"].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(returns.index, returns.values, linewidth=0.7, color="coral")
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title(f"{self.symbol} — Daily Returns")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Return")

        axes[1].hist(returns, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
        axes[1].axvline(returns.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean: {returns.mean():.4f}")
        axes[1].set_title(f"{self.symbol} — Returns Distribution")
        axes[1].set_xlabel("Daily Return")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        fig.tight_layout()
        self._save_or_show(fig, "returns_distribution.png", save)

    def plot_volume(self, save: bool = True) -> None:
        if "Volume" not in self.df.columns:
            logger.warning("Volume column not found. Skipping volume plot.")
            return

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(self.df.index, self.df["Volume"], width=1, color="slategray", alpha=0.75)
        ax.set_title(f"{self.symbol} — Trading Volume")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        self._save_or_show(fig, "volume.png", save)

    def plot_volatility(self, save: bool = True) -> None:
        vol_cols = [c for c in self.df.columns if c.startswith("Volatility_")]
        if not vol_cols:
            logger.warning("No volatility columns found. Skipping volatility plot.")
            return

        fig, ax = plt.subplots(figsize=(14, 4))
        for col in vol_cols:
            ax.plot(self.df.index, self.df[col], linewidth=1.0, label=col)

        ax.set_title(f"{self.symbol} — Rolling Volatility (annualised)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.legend()
        fig.tight_layout()
        self._save_or_show(fig, "volatility.png", save)

    def plot_all(self, save: bool = True) -> None:
        self.plot_closing_price(save)
        self.plot_returns_distribution(save)
        self.plot_volume(save)
        self.plot_volatility(save)

    def _save_or_show(self, fig: plt.Figure, filename: str, save: bool) -> None:
        if save:
            path = self.output_dir / filename
            fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
            logger.info("Plot saved to %s", path)
        else:
            plt.show()
        plt.close(fig)
