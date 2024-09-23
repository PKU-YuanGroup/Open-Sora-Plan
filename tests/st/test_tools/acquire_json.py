import re
import json
import os


def check_is_valid(actual_val, expected_val, margin=0.01, greater=True, message=None):
    cond1 = actual_val > expected_val if greater else actual_val < expected_val
    cond2 = abs(actual_val - expected_val) / expected_val > margin
    if cond1 and cond2:
        if message:
            raise AssertionError(message)
        else:
            raise AssertionError


def transfer_logs_as_json(log_file, output_json_file):
    """
    Read a log file from the input path, and return the
    summary specified as input as a list

    Args:
        log_file: str, path to the dir where the logs are located.
        output_json_file: str, path of the json file transferred from the logs.
    
    Returns:
        data: json, the values parsed from the log, formatted as a json file.
    """
    
    log_pattern = re.compile(
        r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?loss:\s+([0-9.]+E[+-][0-9]+)"
    )

    memory_pattern = re.compile(
        r"\[Rank (\d+)\] \(after \d+ iterations\) memory \(MB\) \| allocated: ([0-9.]+) \| max allocated: ([0-9.]+)"
    )

    data = {
        "loss": [],
        "time": [],
        "memo info": []
    }
    with open(log_file, "r") as f:
        log_content = f.read()

    log_matches = log_pattern.findall(log_content)
    memory_matches = memory_pattern.findall(log_content)

    if log_matches:
        data["loss"] = [float(match[1]) for match in log_matches]
        data["time"] = [float(match[0]) for match in log_matches]

    if memory_matches:
        memo_info = [
            {
                "rank": int(match[0]),
                "allocated memory": float(match[1]),
                "max allocated memory": float(match[2])
            }
            for match in memory_matches
        ]
        data["memo info"] = sorted(memo_info, key=lambda x: x["rank"])

    with open(output_json_file, "w") as outfile:
        json.dump(data, outfile, indent=4)
    os.chmod(output_json_file, 440)


def read_json(file):
    """
    Read baseline and new generate json file
    """
    if os.path.exists(file):
        with open(file) as f:
            return json.load(f)
    else:
        raise FileExistsError("The file does not exist !")
