from unicodedata import name

import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import matplotlib.patches as mpatches

from matplotlib.ticker import MaxNLocator

def merge_jsonl(input_dir="logs"):
    input_path = Path(input_dir)

    kb_files = sorted(input_path.glob("kb_*.jsonl"), reverse=True)
    none_files = sorted(input_path.glob("none_*.jsonl"), reverse=True)
    zs_files = sorted(input_path.glob("zs_*.jsonl"), reverse=True)
    fs_files = sorted(input_path.glob("fs_*.jsonl"), reverse=True)

    def merge(files, output_file):
        if files:
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
    
    # Mapping for your specific naming requirements
    # Ensure these keys match your actual filenames (minus .jsonl)
    name_map = {
        "zs_merged": "Zero-Shot",
        "fs_merged": "Few-Shot",
        "kb_merged": "RAG",
        "none_merged": "Baseline"
    }

    # Load data and rename keys based on the map (fallback to stem if not in map)
    all_data = {name_map.get(f.stem, f.stem): load_metrics(f) for f in files}
    metrics = ["qos", "energy_usage", "fairness", "score", "reward"]

    # ---- Grid subplot (2 rows, 3 columns) ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)) 
    axes = axes.flatten()  

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for name, data in all_data.items():
            y = data[metric]
            ax.plot(range(len(y)), y, label=name)

        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
        ax.set_xlabel("Steps") 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle='--', alpha=0.6)
        # Legend removed from individual plots to keep them clean

    # ---- Setup Legend in the 6th slot ----
    ax_legend = axes[5]
    ax_legend.axis('off') # Hide the plot lines/axes
    
    # Pull the legend handles from one of the existing plots
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Place the legend in the center of the 6th box
    ax_legend.legend(
        handles, 
        labels, 
        loc='center',  
        title_fontsize='16',
        prop={'size': 14}, 
        labelspacing=1.5,
        borderpad=2,
        edgecolor='black',
        facecolor='#f9f9f9'
    )

    plt.tight_layout()
    
    combined_path = output_dir / "comparison_grid.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved grid plot: {combined_path}")

    # ---- Individual plots ----
    for metric in metrics:
        plt.figure()
        for name, data in all_data.items():
            y = data[metric]
            plt.plot(range(len(y)), y, label=name)

        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Steps")
        plt.ylabel(metric.replace("_", " ").title())
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        path = output_dir / f"{metric}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ---- Performance Analysis Grid ---
    _draw_analysis(all_data, output_dir)
    _draw_analysis_scenarios(output_dir)


def _draw_analysis(all_data, output_dir, baseline_name="Baseline"):
    output_dir = Path(output_dir)
    metrics = ["qos", "energy_usage", "fairness", "score"]

    display_names = {
        "qos": "QoS",
        "energy_usage": "Energy Efficiency",
        "fairness": "Fairness",
        "score": "Score"
    }

    methods = list(all_data.keys())
    non_baseline_methods = [m for m in methods if m != baseline_name]

    max_bar_width = 0.15 
    group_spacing = 0.8
    label_fontsize = 15
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
    method_colors = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
    metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig1, ax1 = plt.subplots(figsize=(14, 8)) 
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    fig3, ax3 = plt.subplots(figsize=(14, 8))

    # --- PLOT 1: AVERAGE ABSOLUTE VALUES ---
    x1 = np.arange(len(metrics))
    width1 = min(max_bar_width, group_spacing / len(methods))
    for i, m in enumerate(methods):
        avg_vals = [np.mean(all_data[m].get(met, [0])) for met in metrics]
        offset = (i - len(methods)/2) * width1 + (width1/2)
        bars = ax1.bar(x1 + offset, avg_vals, width1, label=m, color=method_colors[m], edgecolor='black')
        # Add labels on top
        ax1.bar_label(bars, padding=5, fmt='%.3f', fontweight='bold')
    
    ax1.set_title("Absolute Average Performance", fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=label_fontsize)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # --- PLOT 2: AVERAGE % IMPROVEMENT (VS BASELINE) ---
    x2 = np.arange(len(metrics))
    width2 = min(max_bar_width, group_spacing / len(non_baseline_methods)) if non_baseline_methods else max_bar_width
    for i, m in enumerate(non_baseline_methods):
        avg_pct_changes = []
        for metric in metrics:
            is_lower_better = (metric == "energy_usage")

            base = np.array(all_data[baseline_name].get(metric, [0]))
            curr = np.array(all_data[m].get(metric, [0]))

            if len(base) > 0 and len(curr) > 0:
                min_l = min(len(base), len(curr))

                pct = ((curr[:min_l] - base[:min_l]) / (base[:min_l] + 1e-9)) * 100
                avg_diff = np.mean(pct)

                avg_pct_changes.append(-avg_diff if is_lower_better else avg_diff)
            else:
                avg_pct_changes.append(0)
        offset = (i - len(non_baseline_methods)/2) * width2 + (width2/2)
        bars = ax2.bar(
            x2 + offset,
            avg_pct_changes,
            width2,
            label=m,
            color=method_colors[m],
            edgecolor='black'
        )
        # Add labels on top (with % sign)
        label_objs = ax2.bar_label(
            bars,
            padding=10,
            fmt='%.1f',
            fontsize=14,
            fontweight='bold'
        )
        for j, lbl in enumerate(label_objs):
            val = avg_pct_changes[j]
            lbl.set_color('green' if val >= 0 else 'red')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title(
        "Average Improvement (%) vs Baseline",
        fontsize=18,
        fontweight='bold',
        pad=15
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [display_names[m] for m in metrics],
        fontsize=18,
        fontweight='bold'
    )
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_ylabel("Improvement (%)", fontsize=18, fontweight='bold')
    # Legend at top-right corner
    ax2.legend(
        loc='upper right',
        fontsize=18
    )

    # --- PLOT 3: PEAK GAIN & LOSS (%) ---
    x3 = np.arange(len(non_baseline_methods))
    width3 = min(max_bar_width, group_spacing / len(metrics))
    metric_handles = []
    for i, metric in enumerate(metrics):
        gains, losses = [], []
        is_lower_better = (metric == "energy_usage")
        base_vals = np.array(all_data[baseline_name].get(metric, []))

        for m in non_baseline_methods:
            curr_vals = np.array(all_data[m].get(metric, []))
            if len(curr_vals) == 0 or len(base_vals) == 0:
                gains.append(0); losses.append(0)
                continue
            min_l = min(len(curr_vals), len(base_vals))
            pct = ((curr_vals[:min_l] - base_vals[:min_l]) / (base_vals[:min_l] + 1e-9)) * 100
            
            if is_lower_better:
                gains.append(-np.min(pct)); losses.append(-np.max(pct))
            else:
                gains.append(np.max(pct)); losses.append(np.min(pct))

        offset = (i - len(metrics)/2) * width3 + (width3/2)
        
        # Gain Bars
        gbars = ax3.bar(x3 + offset, gains, width3, label=metric.title(), color=metric_colors[i], edgecolor='black')
        ax3.bar_label(gbars, padding=5, fmt='%.1f%%', color='green', fontweight='bold')
        
        # Loss Bars (no labels for these to avoid clutter, but showing the bars)
        lbars = ax3.bar(x3 + offset, losses, width3, color=metric_colors[i], alpha=0.2, hatch='//', edgecolor='black')
        ax3.bar_label(lbars, padding=5, fmt='%.1f%%', color='red', fontweight='bold')

        metric_handles.append(mpatches.Patch(color=metric_colors[i], label=display_names[metric]))

    style_handles = [
        mpatches.Patch(facecolor='gray', edgecolor='black', label='Peak Gain (Solid)'),
        mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.3, hatch='//', label='Max Loss (Shaded)')
    ]

    # Create one unified legend to prevent overflow
    ax3.legend(handles=metric_handles + style_handles, 
            loc='upper left', 
            bbox_to_anchor=(1, 1),
            fontsize=9)

    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_title("Peak Gain vs Max Loss (%)", fontsize=15, fontweight='bold', pad=15)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(non_baseline_methods, fontsize=label_fontsize)

    if len(non_baseline_methods) == 1:
        ax3.set_xlim(-0.8, 0.8)

    for ax in [ax1, ax2, ax3]:
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        # Increase top margin so labels don't get cut off
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.15)
        ax.tick_params(axis='y', labelsize=label_fontsize)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()        
    fig1.savefig(output_dir / "absolute_values.png", dpi=300, bbox_inches='tight')
    print(f"Saved performance analysis: {output_dir / 'absolute_values.png'}")
    fig2.savefig(output_dir / "avg_improvement.png", dpi=300, bbox_inches='tight')
    print(f"Saved performance analysis: {output_dir / 'avg_improvement.png'}")
    fig3.savefig(output_dir / "peak_range.png", dpi=300, bbox_inches='tight')
    print(f"Saved performance analysis: {output_dir / 'peak_range.png'}")
    
    img1 = plt.imread(output_dir / "absolute_values.png")
    img2 = plt.imread(output_dir / "avg_improvement.png")
    img3 = plt.imread(output_dir / "peak_range.png")
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))

    axs[0].imshow(img1)
    axs[1].imshow(img2)
    axs[2].imshow(img3)

    for ax in axs:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "summary_analysis.png", dpi=300)
    print(f"Saved summary performance analysis: {output_dir / 'summary_analysis.png'}")

def _draw_analysis_scenario(jsonl_path, output_dir="eval_plots"):
    data = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            method = row["metadata"]

            data[method] = {
                "qos": row["qos"],
                "energy_usage": row["energy_usage"],
                "fairness": row["fairness"],
                "score": row["score"]
            }
    
    title_map = {
        "mean": "Mean Pressure Scenario",
        "high_pressure": "High Pressure Scenario",
        "low_pressure": "Low Pressure Scenario"
    }
    scenario = Path(jsonl_path).stem
    plot_title = title_map.get(scenario, scenario.title())

    metrics = ["qos", "energy_usage", "fairness", "score"]

    display_names = {
        "qos": "QoS",
        "energy_usage": "Energy Efficiency",
        "fairness": "Fairness",
        "score": "Score"
    }

    baseline = data["Baseline"]

    methods = ["RAG", "Few-shot", "Zero-shot"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, method in enumerate(methods):

        pct_changes = []

        for metric in metrics:

            base = baseline[metric]
            curr = data[method][metric]

            pct = ((curr - base) / (base + 1e-9)) * 100

            # lower is better
            if metric == "energy_usage":
                pct = -pct

            pct_changes.append(pct)

        offset = (i - len(methods)/2) * width + width/2

        bars = ax.bar(
            x + offset,
            pct_changes,
            width,
            label=method
        )

        labels = ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=3,
            fontsize=15
        )

        for j, lbl in enumerate(labels):
            lbl.set_color(
                'green' if pct_changes[j] >= 0 else 'red'
            )

    ax.axhline(0, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_names[m] for m in metrics],
        fontsize=15
    )

    ax.tick_params(axis='y', labelsize=15)
    ax.set_ylabel("Gain (%)", fontsize=15)
    ax.set_title(
        plot_title,
        fontsize=18,
        fontweight='bold'
    )

    ax.legend(fontsize=15)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(
        ymin - abs(ymin) * 0.2,
        ymax + abs(ymax) * 0.2
    )
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{scenario}.png",
        dpi=300,
        bbox_inches='tight'
    )
    print(f"Saved {scenario} performance analysis: {output_dir / f'{scenario}.png'}")

def _draw_analysis_scenarios(output_dir="eval_plots"):
    paths = [
        "logs/mean.jsonl",
        "logs/high_pressure.jsonl",
        "logs/low_pressure.jsonl"
    ]
    for path in paths:
        _draw_analysis_scenario(path, output_dir)

if __name__ == "__main__":
    import time
    merge_jsonl()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    plot_and_save(output_dir=f"plots_{stamp}")