import pandas as pd
import matplotlib.pyplot as plt
import imageio
import glob

frames = []

csv_files = sorted(
    glob.glob("../results/particles_step_*.csv"),
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

if len(csv_files) == 0:
    print("No CSV files found in ../results/")
    print("Run the C++ simulation first: cd ../cpp && ./sim")
    exit(1)

print(f"Found {len(csv_files)} CSV files, creating animation...")

for csv in csv_files:
    df = pd.read_csv(csv)
    plt.figure(figsize=(5,5))
    plt.scatter(df["x"], df["y"], s=df["mass"]*30, alpha=0.7)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.title(f"Step {df['step'].iloc[0]}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.savefig("frame.png", dpi=100, bbox_inches='tight')
    plt.close()
    frames.append(imageio.imread("frame.png"))

imageio.mimsave("../results/simulation.gif", frames, fps=4)
print(f"GIF saved: ../results/simulation.gif ({len(frames)} frames)")