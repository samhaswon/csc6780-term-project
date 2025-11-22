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


def parse_config_for_model_and_nodes(config_path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse base_session model and number of nodes from config.yml, using regex.

    Looks for:
      base_session:
        - "MODEL_NAME"

    And:
      patch_servers:
        - "nodeA:5432"
        - "nodeB:5432"

    :param config_path: Path to config.yml file.
    :returns: (base_session_model, num_nodes) or (None, None) on failure.
    """
    if not os.path.exists(config_path):
        return None, None

    with open(config_path, "r", encoding="utf-8", errors="ignore") as cfg:
        text = cfg.read()

    # base_session: first list entry
    base_session = None
    base_match = re.search(
        r"base_session:\s*\n\s*-\s*['\"]([^'\"]+)['\"]",
        text,
        flags=re.MULTILINE,
    )
    if base_match:
        base_session = base_match.group(1)

    # patch_servers: count the list entries
    num_nodes = None
    patch_section_match = re.search(
        r"patch_servers:\s*((?:\n\s*-\s*['\"][^'\"]+['\"])+)",
        text,
        flags=re.MULTILINE,
    )
    if patch_section_match:
        block = patch_section_match.group(1)
        # each list entry starts with "-"
        num_nodes = len(re.findall(r"\n\s*-\s*['\"][^'\"]+['\"]", block))

    return base_session, num_nodes


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

    Looks under ./runs, finds latest manager-*.out, parses timings and config,
    and appends rows to ./runs/manager_timings.csv.
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

    config_path = os.path.join(run_dir, "config.yml")
    base_session, num_nodes = parse_config_for_model_and_nodes(config_path)

    print(f"Run dir: {run_dir}")
    print(f"Config: {config_path}")
    print(f"base_session: {base_session}")
    print(f"nodes: {num_nodes}")

    timings = parse_timings_from_log(manager_log)
    print(f"Found {len(timings)} timing blocks in log")

    append_timings_to_csv(csv_path, base_session, num_nodes, timings)
    print(f"Appended timings to {csv_path}")


if __name__ == "__main__":
    main()
