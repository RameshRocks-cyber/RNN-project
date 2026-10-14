"""
benchmark_charts.py
Generate professional comparison charts comparing SimpleRNN, LSTM, and GRU models.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

def generate_comparison_dashboard(output_path: str):
    """Generates a 4-panel comparison dashboard across SimpleRNN, LSTM, and GRU."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    models = ["SimpleRNN", "LSTM", "GRU"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    
    # 1. Empirical Test Metrics
    accuracies = [49.97, 87.58, 88.26]
    precisions = [49.97, 86.85, 89.19]
    recalls = [100.00, 88.63, 87.18]
    f1_scores = [66.64, 87.73, 88.18]
    
    # 2. Computational Metrics
    total_params = [650369, 675137, 667073]
    recurrent_params = [8256, 33024, 24960]
    training_times = [113.88, 136.94, 119.61]
    best_val_losses = [0.6934, 0.3126, 0.2761]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    # -------------------------------------------------------------
    # Plot 1: Test Performance Comparison (Grouped Bar Chart)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.18
    
    b1 = ax1.bar(x - 1.5*width, accuracies, width, label="Accuracy", color="#34495e", alpha=0.9)
    b2 = ax1.bar(x - 0.5*width, precisions, width, label="Precision", color="#3498db", alpha=0.9)
    b3 = ax1.bar(x + 0.5*width, recalls, width, label="Recall", color="#e67e22", alpha=0.9)
    b4 = ax1.bar(x + 1.5*width, f1_scores, width, label="F1-Score", color="#2ecc71", alpha=0.9)
    
    ax1.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Test Set Performance Metrics Comparison", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    
    # Add values on top of bars
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, rotation=90)
    
    # -------------------------------------------------------------
    # Plot 2: Recurrent vs Total Parameter Footprint
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    x_pos = np.arange(len(models))
    
    # Recurrent params
    bars_rec = ax2.bar(x_pos - 0.18, [p / 1000 for p in recurrent_params], 0.35, 
                       label="Recurrent Layer Params (k)", color=["#e74c3c", "#2980b9", "#27ae60"], alpha=0.85)
    
    # Total params (secondary y-axis)
    ax2_twin = ax2.twinx()
    bars_total = ax2_twin.bar(x_pos + 0.18, [p / 1000 for p in total_params], 0.35, 
                             label="Total Architecture Params (k)", color="#7f8c8d", alpha=0.6)
    
    ax2.set_ylabel("Recurrent Params (Thousands)", fontsize=11, fontweight="bold", color="#2c3e50")
    ax2_twin.set_ylabel("Total Params (Thousands)", fontsize=11, fontweight="bold", color="#7f8c8d")
    ax2.set_title("Parameter Footprint (Recurrent vs Total)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 42)
    ax2_twin.set_ylim(600, 700)
    ax2.grid(False)
    ax2_twin.grid(True, linestyle="--", alpha=0.4)
    
    for bar in bars_rec:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}k", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_total:
        h = bar.get_height()
        ax2_twin.annotate(f"{h:.1f}k", xy=(bar.get_x() + bar.get_width()/2, h),
                          xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Plot 3: Training Wall-Clock Time vs Validation Loss
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    bar_time = ax3.bar(x_pos - 0.18, training_times, 0.35, color=colors, alpha=0.85, label="Training Time (s)")
    ax3.set_ylabel("Wall-Clock Time (Seconds)", fontsize=11, fontweight="bold")
    ax3.set_title("Training Duration & Validation Loss Comparison", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 165)
    
    for bar in bar_time:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f} s", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax3_twin = ax3.twinx()
    line_loss = ax3_twin.plot(x_pos, best_val_losses, color="#8e44ad", marker="o", linewidth=2.5, 
                             markersize=8, label="Best Validation Loss")
    ax3_twin.set_ylabel("Best Validation Loss (Binary Crossentropy)", fontsize=11, fontweight="bold", color="#8e44ad")
    ax3_twin.set_ylim(0.2, 0.75)
    ax3.grid(False)
    ax3_twin.grid(True, linestyle="--", alpha=0.4)
    
    for x_i, y_i in zip(x_pos, best_val_losses):
        ax3_twin.annotate(f"{y_i:.4f}", xy=(x_i, y_i), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#8e44ad")
        
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Plot 4: Pareto Efficiency Frontier (Accuracy vs Training Time)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    sizes = [p / 300 for p in recurrent_params]  # Bubble size represents recurrent parameter size
    
    for i, model in enumerate(models):
        scatter = ax4.scatter(training_times[i], accuracies[i], s=sizes[i]*8, color=colors[i], 
                              alpha=0.75, edgecolors="black", linewidth=1.5, label=f"{model} (Recurrent Params: {recurrent_params[i]:,})")
        offset_y = 1.2 if model != "SimpleRNN" else -3.5
        ax4.annotate(f"{model}\n({accuracies[i]:.2f}%, {training_times[i]:.1f}s)",
                     xy=(training_times[i], accuracies[i]),
                     xytext=(0, offset_y * 8), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], lw=1.5, alpha=0.9))
        
    ax4.set_xlabel("Training Duration (Seconds)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax4.set_title("Pareto Efficiency: Accuracy vs Training Latency", fontsize=13, fontweight="bold", pad=12)
    ax4.set_xlim(100, 150)
    ax4.set_ylim(40, 95)
    ax4.axhline(50, color="gray", linestyle=":", alpha=0.7, label="Random Guess / Degenerate Baseline")
    ax4.legend(loc="lower right", frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Benchmark comparison dashboard successfully saved to: {output_path}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    chart_file = os.path.join(out_dir, "model_comparison_charts.png")
    generate_comparison_dashboard(chart_file)
