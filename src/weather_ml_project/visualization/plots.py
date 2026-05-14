from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_leakage_matrix(df, output: Path) -> None:
    corr = df.corr().abs()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Matrice de corrélation pour détecter le leakage potentiel")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


def plot_train_val_split(train_df, val_df, output: Path) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(train_df["datetime"], train_df["temperature"], label="Train temperature", alpha=0.5)
    plt.plot(val_df["datetime"], val_df["temperature"], label="Val temperature", alpha=0.7)
    plt.legend()
    plt.title("Split chronologique train/validation")
    plt.xlabel("Datetime")
    plt.ylabel("Temperature")
    plt.savefig(output, dpi=150)
    plt.close()
