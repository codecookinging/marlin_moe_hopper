import os
import subprocess
import json
import tempfile
import sys
import re
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Search for the best grid size across GPUs.")
parser.add_argument("--case", type=str, default="glm5_prefill_m256", help="Specific case to run (default: glm5_prefill_m256).")
parser.add_argument("--all-cases", action="store_true", help="Run all built-in cases. If set, --case is ignored.")
args = parser.parse_args()

# Candidate grids to test
# We test various multiples up to SMS * BMAX (132 * 6 = 792)
candidates = list(range(8, 792, 8))
candidates = sorted(list(set(candidates)))

print("Step 1: Skipping recompilation (assuming already built)...")

if args.all_cases:
    print(f"\nStep 2: Benchmarking {len(candidates)} candidate grids for ALL cases across 8 GPUs...")
else:
    print(f"\nStep 2: Benchmarking {len(candidates)} candidate grids for case '{args.case}' across 8 GPUs...")

def run_chunk(gpu_id, chunk):
    chunk_results = []
    for grid in chunk:
        env = os.environ.copy()
        env["MARLIN_MOE_FORCE_GRID"] = str(grid)
        env["PYTHONPATH"] = f"{os.getcwd()}/python"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_json = tmp.name

        cmd = [
            sys.executable, "benchmark_marlin_moe_ab.py", 
            "--mode", "single", 
            "--timing", "batch", 
            "--output-json", tmp_json
        ]
        
        if not args.all_cases:
            cmd.extend(["--cases", args.case])
        
        grid_times = {}
        try:
            # Run the benchmark
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Read the JSON output
            if os.path.exists(tmp_json) and os.path.getsize(tmp_json) > 0:
                with open(tmp_json, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for record in data:
                            # Extract case name
                            case_name = record.get("case", {}).get("name", "unknown_case")
                            
                            # Extract time
                            time_ms = None
                            for key in ['mean_us', 'p50_us', 'total_us', 'time_ms', 'mean_ms', 'batch_ms', 'latency_ms']:
                                if key in record:
                                    val = float(record[key])
                                    if key.endswith('_us'):
                                        val /= 1000.0  # Convert us to ms
                                    time_ms = val
                                    break
                            
                            if time_ms is not None:
                                grid_times[case_name] = time_ms
            
            # Fallback to parsing stdout if JSON parsing failed (only works reliably for single case)
            if not grid_times and not args.all_cases:
                for line in proc.stdout.split('\n'):
                    if args.case in line and 'bfloat16' not in line:
                        numbers = re.findall(r"[\d.]+", line)
                        grid_times[args.case] = float(numbers[-2])
                        break
                            
            if grid_times:
                chunk_results.append((grid, grid_times))
                # Format a short summary for the console
                cases_str = ", ".join([f"{k}: {v:.4f}ms" for k, v in grid_times.items()])
                if len(cases_str) > 80:
                    cases_str = cases_str[:77] + "..."
                print(f"[GPU {gpu_id}] Grid {grid:3d} -> {cases_str}")
            else:
                print(f"[GPU {gpu_id}] Grid {grid:3d} -> Failed to parse time.")
                
        except Exception as e:
            print(f"[GPU {gpu_id}] Grid {grid:3d} -> Error: {e}")
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
                
    return chunk_results

# Split candidates into 8 chunks
num_gpus = 8
chunks = [candidates[i::num_gpus] for i in range(num_gpus)]

all_results = []
with ThreadPoolExecutor(max_workers=num_gpus) as executor:
    futures = [executor.submit(run_chunk, i, chunks[i]) for i in range(num_gpus)]
    for future in futures:
        all_results.extend(future.result())

print("\nStep 3: Results Summary")
if not all_results:
    print("No valid results collected. Please check the benchmark script output.")
    sys.exit(1)

# Reorganize results by case: { case_name: [(grid1, time1), (grid2, time2), ...] }
case_results = defaultdict(list)
for grid, times in all_results:
    for case_name, t in times.items():
        case_results[case_name].append((grid, t))

output_file = "grid_search_results.txt"
with open(output_file, "w") as f:
    for case_name, results in case_results.items():
        # Sort by execution time (ascending)
        results.sort(key=lambda x: x[1])
        
        header = f"\n=== Results for Case: {case_name} ===\n"
        f.write(header)
        print(header, end="")
        
        table_border = "+" + "-"*30 + "+\n"
        f.write(table_border)
        print(table_border, end="")
        
        row_header = "| Grid Size | Execution Time (ms) |\n"
        f.write(row_header)
        print(row_header, end="")
        
        f.write(table_border)
        print(table_border, end="")
        
        for grid, time_ms in results:
            if grid == results[0][0]:
                line = f"| {grid:9d} | {time_ms:17.4f} * |  <-- BEST\n"
            else:
                line = f"| {grid:9d} | {time_ms:17.4f}   |\n"
            f.write(line)
            print(line, end="")
            
        f.write(table_border)
        print(table_border, end="")
        
        best_grid = results[0][0]
        best_time = results[0][1]
        summary = f"Optimal Grid for {case_name} is {best_grid} ({best_time:.4f} ms).\n"
        f.write(summary)
        print(summary, end="")

print(f"\nAll results saved to {output_file}.")
