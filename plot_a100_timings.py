from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_timing_results(csv_path: str):
    path = Path(csv_path)

    if not path.exists():
        print(f"Error: File {path} not found.")
        return

    # 2. Read the CSV file using Pandas
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Verify columns exist
    required_cols = ["base", "patch", "total"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV is missing one of the required columns: {required_cols}")
        return

    # 3. Plotting
    averages = df[required_cols].mean()

    plt.figure(figsize=(8, 6))
    plt.style.use('ggplot')  # Use a nice style

    # Plot bars for average metrics
    colors = ['#E24A33', '#348ABD', '#333333']  # matches base, patch, total
    bars = plt.bar(averages.index, averages.values, color=colors)

    # 4. Styling the chart
    plt.title(f'Average Timing Results (Dual A100)', fontsize=14, pad=15)
    plt.ylabel('Average Time (seconds)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}s',
                 ha='center', va='bottom')

    plt.tight_layout()

    # Show plot
    plt.savefig("./plots/a100_timings.png")
    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    plot_timing_results("./a100_test.csv")