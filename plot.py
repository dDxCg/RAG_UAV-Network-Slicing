import matplotlib.pyplot as plt
from pathlib import Path
import json

from matplotlib.ticker import MaxNLocator

def merge_jsonl(input_dir="logs"):
    input_path = Path(input_dir)

    kb_files = sorted(input_path.glob("kb_*.jsonl"), reverse=True)
    none_files = sorted(input_path.glob("none_*.jsonl"), reverse=True)
    zs_files = sorted(input_path.glob("zs_*.jsonl"), reverse=True)
    fs_files = sorted(input_path.glob("fs_*.jsonl"), reverse=True)

    def merge(files, output_file):
        with open(output_file, "w", encoding="utf-8") as outfile:
            for file in files:
                with open(file, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)

    merge(kb_files, f"{input_dir}/kb_merged.jsonl")
    merge(none_files, f"{input_dir}/none_merged.jsonl")
    merge(zs_files, f"{input_dir}/zs_merged.jsonl")
    merge(fs_files, f"{input_dir}/fs_merged.jsonl")

    print(f"Merged {len(kb_files)} kb files → kb_merged.jsonl")
    print(f"Merged {len(none_files)} none files → none_merged.jsonl")
    print(f"Merged {len(zs_files)} zs files → zs_merged.jsonl")
    print(f"Merged {len(fs_files)} fs files → fs_merged.jsonl")



def load_metrics(file_path):
    qos, energy, fairness, score, reward = [], [], [], [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            qos.append(data.get("qos", 0))
            energy.append(data.get("energy_usage", 0))
            fairness.append(data.get("fairness", 0))
            score.append(data.get("score", 0))
            reward.append(data.get("reward", 0))

    return {
        "qos": qos,
        "energy_usage": energy,
        "fairness": fairness,
        "score": score,
        "reward": reward
    }


def plot_and_save(folder="logs", output_dir="logs"):
    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(folder.glob("*_merged.jsonl"))
    all_data = {f.stem: load_metrics(f) for f in files}

    metrics = ["qos", "energy_usage", "fairness", "score", "reward"]

    # ---- Grid subplot (2 rows, 3 columns) ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)) 
    axes = axes.flatten()  

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for name, data in all_data.items():
            y = data[metric]
            x = range(len(y))
            ax.plot(x, y, label=name)

        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
        ax.set_xlabel("Steps") 

        # Force the x-axis to only use integers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax.legend(fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)

    if len(metrics) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    
    combined_path = output_dir / "comparison_grid.png"
    plt.savefig(combined_path)
    plt.close()
    print(f"Saved grid plot: {combined_path}")

    for metric in metrics:
        plt.figure()
        for name, data in all_data.items():
            y = data[metric]
            plt.plot(range(len(y)), y, label=name)

        plt.title(metric)
        plt.xlabel("Steps")
        plt.ylabel(metric)

        # Force the x-axis to only use integers
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.legend()
        plt.tight_layout()

        path = output_dir / f"{metric}.png"
        plt.savefig(path)
        plt.close()
        print(f"Saved: {path}")


if __name__ == "__main__":
    import time
    merge_jsonl()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    plot_and_save(output_dir=f"plots_{stamp}")