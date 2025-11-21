import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_snapshot(csv_path: str, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    step = df["step"].iloc[0] if "step" in df else "unknown"

    plt.figure(figsize=(6, 6))
    mass_col = df["mass"] if "mass" in df.columns else 1.0
    plt.scatter(df["x"], df["y"], s=mass_col * 50, alpha=0.6)
    plt.title(f'Particle Positions at Step {step}')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(output_dir, f"{base}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot particle positions from CSV files")
    parser.add_argument("--csv", help="Path to CSV file to plot")
    args = parser.parse_args()
    
    if args.csv:
        plot_snapshot(args.csv)
    else:
        print("Plot particle positions from Monte Carlo simulation CSV files.")
        print("Usage: python plot_results.py --csv path/to/particles_step_XXX.csv")
        print("")
        print("To create animations, use: python make_gif.py")
