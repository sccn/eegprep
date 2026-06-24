import json
import sys
import os
from pathlib import Path

def main():
    matrix_path = Path("docs/parity/eeglab_core_parity_matrix.json")
    metrics_path = Path(".parity_metrics.json")
    
    if not matrix_path.exists():
        print(f"Matrix not found at {matrix_path}")
        return 1
        
    if not metrics_path.exists():
        print("No parity metrics found.")
        return 0
        
    with open(matrix_path, "r", encoding="utf-8") as f:
        matrix = json.load(f)
        
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    has_regression = False
    updated = False
    
    for row in matrix.get("rows", []):
        func_name = row.get("eeglab_name")
        if func_name in metrics:
            new_m = metrics[func_name]
            
            # Get configurable thresholds
            thresholds = row.get("thresholds", {})
            max_allowed_rms = thresholds.get("rms_diff", 1e-5)
            max_allowed_max = thresholds.get("max_diff", 1e-5)
            
            old_m = row.get("metrics", {})
            
            rms_exceeded = new_m["rms_diff"] > max_allowed_rms
            max_exceeded = new_m["max_diff"] > max_allowed_max
            
            # Did it regress compared to what was previously recorded?
            if old_m:
                if new_m["rms_diff"] > old_m.get("rms_diff", max_allowed_rms) * 1.5:
                    rms_exceeded = True
                if new_m["max_diff"] > old_m.get("max_diff", max_allowed_max) * 1.5:
                    max_exceeded = True
                    
            if rms_exceeded or max_exceeded:
                if row.get("status") == "implemented":
                    state = "Regressed"
                    has_regression = True
                else:
                    state = "In Progress"
                
                print(f"[{state}] {func_name}: RMS={new_m['rms_diff']} (limit={max_allowed_rms}), Max={new_m['max_diff']} (limit={max_allowed_max})")
                
                # If it's in progress, we can still record its metrics if it improved
                if state == "In Progress":
                    if not old_m or new_m["rms_diff"] < old_m.get("rms_diff", float('inf')) or new_m["max_diff"] < old_m.get("max_diff", float('inf')):
                        row["metrics"] = new_m
                        updated = True
            else:
                state = "Verified Parity"
                print(f"[{state}] {func_name}: RMS={new_m['rms_diff']}, Max={new_m['max_diff']}")
                # Only update if improved or not recorded
                if new_m["rms_diff"] < old_m.get("rms_diff", float('inf')) or new_m["max_diff"] < old_m.get("max_diff", float('inf')) or not old_m:
                    row["metrics"] = new_m
                    row["status"] = "implemented"
                    updated = True
                else:
                    row["metrics"] = new_m
                    updated = True
                    
    if updated:
        with open(matrix_path, "w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2)
            f.write("\n")
            
    if has_regression:
        print("Build failed due to parity regressions.")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
