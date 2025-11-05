import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_log_history(trainer_state_path: Path):
    with open(trainer_state_path, "r") as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    df = pd.DataFrame(logs)

    # Convert columns to numeric when possible
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    # Ensure step column exists
    if "step" not in df.columns and len(df) > 0:
        df["step"] = range(1, len(df) + 1)

    return state, df


def rolling_series(s, window=5):
    try:
        return s.rolling(window=window, min_periods=max(1, window//2)).median()
    except Exception:
        return s


def savefig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    print(f"✅ Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Pretty plots from HuggingFace trainer_state.json")
    parser.add_argument("--trainer_state", type=str, default="trainer_state.json", help="Path to trainer_state.json")
    parser.add_argument("--out_dir", type=str, default="plots", help="Output directory for figures")
    parser.add_argument("--rolling", type=int, default=5, help="Rolling median window")
    args = parser.parse_args()

    state, df = load_log_history(Path(args.trainer_state))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
    df_eval = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()

    # 1) Training loss
    if not df_train.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_train["step"], df_train["loss"], linewidth=1, alpha=0.6, label="train loss")
        ax.plot(df_train["step"], rolling_series(df_train["loss"], args.rolling), linewidth=2, label=f"rolling loss ({args.rolling})")
        ax.set_title("Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.4)
        ax.legend()
        savefig(fig, out_dir, "01_train_loss")

    # 2) Validation loss
    if not df_eval.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_eval["step"], df_eval["eval_loss"], marker="o", linestyle="--", label="eval loss")
        ax.set_title("Validation Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.4)
        ax.legend()
        savefig(fig, out_dir, "02_validation_loss")

    # 3) Learning rate
    if "learning_rate" in df.columns and df["learning_rate"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["step"], df["learning_rate"], linewidth=2)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Step")
        ax.set_ylabel("LR")
        ax.grid(alpha=0.4)
        savefig(fig, out_dir, "03_learning_rate")

    # 5) Perplexity (derived from eval loss)
    if not df_eval.empty:
        eval_loss = df_eval["eval_loss"].to_numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            ppl = np.exp(eval_loss)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_eval["step"], ppl, marker="o", linestyle="-")
        ax.set_title("Validation Perplexity")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.grid(alpha=0.4)
        savefig(fig, out_dir, "05_perplexity")

    print("\n✨ Done. PNG plots saved in:", out_dir.resolve())


if __name__ == "__main__":
    main()
