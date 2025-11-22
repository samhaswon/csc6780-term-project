#!/usr/bin/env python3
import csv
import glob
import os
import re
from typing import List, Tuple, Optional


def find_latest_manager_log(runs_root: str) -> Optional[str]:
    """
    Find the newest manager-*.out file under runs_root.

    :param runs_root: Root directory containing run logs.
    :returns: Path to newest manager log or None if none found.
    """
    pattern = os.path.join(runs_root, "**", "manager-*.out")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def extract_run_dir_from_log(log_path: str) -> Optional[str]:
    """
    Extract the run directory from a manager log.

    Assumes a line like: "Run dir: /path/to/run-123456789".

    :param log_path: Path to manager log.
    :returns: Run directory path or None if not found.
    """
    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        for line in log_file:
            if line.startswith("Run dir: "):
                return line.strip().split("Run dir: ", 1)[1]
    return None


def parse_run_pipeline_for_model_and_nodes(
    pipeline_path: str,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse base_session model and number of nodes from run_pipeline.sh.

    Looks for something like:
        #SBATCH --nodes=4              # Set the number of worker nodes for this run

    And:
        echo 'base_session:'
        echo '  - "birefnet"'          # u2net, u2netp, or birefnet

    :param pipeline_path: Path to run_pipeline.sh file.
    :returns: (base_session_model, num_nodes) or (None, None) on failure.
    """
    if not os.path.exists(pipeline_path):
        return None, None

    with open(pipeline_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Number of nodes: look for #SBATCH --nodes=<N>
    nodes = None
    nodes_match = re.search(r"#SBATCH\s+--nodes=(\d+)", text)
    if nodes_match:
        try:
            nodes = int(nodes_match.group(1))
        except ValueError:
            nodes = None

    # base_session model: look for echo 'base_session:' then the next echo with the model
    model = None
    model_match = re.search(
        r"echo 'base_session:'\s*\n\s*echo '  - \"([^\"]+)\"'",
        text,
        flags=re.MULTILINE,
    )
    if model_match:
        model = model_match.group(1)

    return model, nodes


def parse_timings_from_log(log_path: str) -> List[Tuple[float, float, float]]:
    """
    Parse all Base/Patch/Total timing blocks from a manager log.

    Assumes blocks like:
        Base time: 0.1234s
        Patch time: 0.5678s
        Total time: 0.9999s

    :param log_path: Path to manager log.
    :returns: List of (base_time, patch_time, total_time).
    """
    timings: List[Tuple[float, float, float]] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        lines = list(log_file)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        base_match = re.search(r"Base time:\s*([0-9.]+)s", line)
        if not base_match:
            i += 1
            continue

        if i + 2 >= len(lines):
            break

        patch_line = lines[i + 1].strip()
        total_line = lines[i + 2].strip()

        patch_match = re.search(r"Patch time:\s*([0-9.]+)s", patch_line)
        total_match = re.search(r"Total time:\s*([0-9.]+)s", total_line)

        if not patch_match or not total_match:
            i += 1
            continue

        base_time = float(base_match.group(1))
        patch_time = float(patch_match.group(1))
        total_time = float(total_match.group(1))

        timings.append((base_time, patch_time, total_time))
        i += 3

    return timings


def append_timings_to_csv(
    csv_path: str,
    base_session: Optional[str],
    num_nodes: Optional[int],
    timings: List[Tuple[float, float, float]],
) -> None:
    """
    Append timing rows to a CSV file.

    Drops the first timing entry before writing.

    :param csv_path: Path to CSV file.
    :param base_session: Model name from base_session.
    :param num_nodes: Number of nodes used in the run.
    :param timings: List of (base_time, patch_time, total_time).
    """
    if not timings:
        print("No timing blocks found, nothing to write.")
        return

    # Drop the first instance
    timings = timings[1:]
    if not timings:
        print("Only one timing block found, nothing to write after dropping first.")
        return

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        if not file_exists:
            writer.writerow(
                ["base_session", "nodes", "base_time", "patch_time", "total_time"]
            )

        for base_time, patch_time, total_time in timings:
            writer.writerow(
                [
                    base_session if base_session is not None else "",
                    num_nodes if num_nodes is not None else "",
                    base_time,
                    patch_time,
                    total_time,
                ]
            )


def main() -> None:
    """
    Main entry point.

    Looks under ./runs, finds latest manager-*.out, parses timings, and
    pulls model/nodes from run_pipeline.sh. Appends rows to ./runs/manager_timings.csv.
    """
    runs_root = "./runs"
    csv_path = "./manager_timings.csv"

    manager_log = find_latest_manager_log(runs_root)
    if manager_log is None:
        print("No manager-*.out logs found under ./runs")
        return

    print(f"Using latest manager log: {manager_log}")

    run_dir = extract_run_dir_from_log(manager_log)
    if run_dir is None:
        print("Could not find 'Run dir: ...' in manager log")
        return

    print(f"Run dir: {run_dir}")

    pipeline_path = "./run_pipeline.sh"
    base_session, num_nodes = parse_run_pipeline_for_model_and_nodes(pipeline_path)

    print(f"Parsed from {pipeline_path}:")
    print(f"  base_session: {base_session}")
    print(f"  nodes: {num_nodes}")

    timings = parse_timings_from_log(manager_log)
    print(f"Found {len(timings)} timing blocks in log")

    append_timings_to_csv(csv_path, base_session, num_nodes, timings)
    print(f"Appended timings to {csv_path}")


if __name__ == "__main__":
    main()
