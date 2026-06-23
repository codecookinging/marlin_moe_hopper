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

def get_all_cases():
    cases = ['glm5_decode_m1', 'glm5_decode_m2', 'glm5_decode_m4', 'glm5_decode_m8', 'glm5_prefill_m1024', 'glm5_prefill_m128', 'glm5_prefill_m2048', 'glm5_prefill_m256', 'glm5_prefill_m4096', 'glm5_prefill_m512', 'glm5_prefill_m8192', 'glm5_small_m16', 'glm5_small_m32', 'glm5_small_m64']
    return cases

if args.all_cases:
    target_cases = get_all_cases()
    if not target_cases:
        print("Failed to parse cases from --list-cases.")
        sys.exit(1)
else:
    target_cases = [args.case]

# Candidate grids to test
# We test various multiples up to SMS * BMAX (132 * 6 = 792)
candidates = list(range(8, 792, 8))
candidates = sorted(list(set(candidates)))

print("Step 1: Skipping recompilation (assuming already built)...")
print(f"\nStep 2: Benchmarking {len(candidates)} candidate grids for {len(target_cases)} cases across 8 GPUs...")

# Generate all (grid, case) combinations
combinations = [(g, c) for g in candidates for c in target_cases]

# Split combinations into 8 chunks
num_gpus = 8
chunks = [combinations[i::num_gpus] for i in range(num_gpus)]

def run_chunk(gpu_id, chunk):
    # Group by grid to minimize subprocess calls
    grid_to_cases = defaultdict(list)
    for g, c in chunk:
        grid_to_cases[g].append(c)
        
    chunk_results = []
    
    for grid, cases_for_grid in grid_to_cases.items():
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
            "--output-json", tmp_json,
            "--cases"
        ] + cases_for_grid
        
        try:
            # Run the benchmark
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Read the JSON output
            parsed_count = 0
            if os.path.exists(tmp_json) and os.path.getsize(tmp_json) > 0:
                with open(tmp_json, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for record in data:
                            case_name = record.get("case", {}).get("name", "unknown_case")
                            time_ms = None
                            for key in ['mean_us', 'p50_us', 'total_us', 'time_ms', 'mean_ms', 'batch_ms', 'latency_ms']:
                                if key in record:
                                    val = float(record[key])
                                    if key.endswith('_us'):
                                        val /= 1000.0  # Convert us to ms
                                    time_ms = val
                                    break
                            
                            if time_ms is not None:
                                chunk_results.append((grid, case_name, time_ms))
                                parsed_count += 1
            
            # Fallback to parsing stdout if JSON parsing failed (only works reliably for single case)
            if parsed_count == 0:  #这里 后面只设置单一目标，目前多目标 提取时间
                single_case = cases_for_grid[0]
                for line in proc.stdout.split('\n'):
                    if single_case in line and 'bfloat16' not in line:
                        numbers = re.findall(r"[\d.]+", line)
                        time_ms = float(numbers[-2])
                        chunk_results.append((grid, single_case, time_ms))
                        parsed_count += 1
                        break
                            
            if parsed_count > 0:
                print(f"[GPU {gpu_id}] Grid {grid:3d} -> Processed {parsed_count}/{len(cases_for_grid)} cases.")
            else:
                print(f"[GPU {gpu_id}] Grid {grid:3d} -> Failed to parse time.")
                
        except Exception as e:
            print(f"[GPU {gpu_id}] Grid {grid:3d} -> Error: {e}")
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
                
    return chunk_results

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
for grid, case_name, time_ms in all_results:
    case_results[case_name].append((grid, time_ms))

output_file = "grid_search_results.txt"
with open(output_file, "w") as f:
    for case_name in target_cases:
        if case_name not in case_results:
            continue
            
        results = case_results[case_name]
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
        
        # Print top 5 and the specific grid if needed, but let's just print all or top 10 to save space
        # Actually, let's print all so user has the full lookup table
        for grid, time_ms in results:
            if grid == results[0][0]:
                line = f"| {grid:9d} | {time_ms:17.4f} * |  <-- BEST\n"
            else:
                line = f"| {grid:9d} | {time_ms:17.4f}   |\n"
            f.write(line)
            # Only print top 3 to console to avoid spamming, but write all to file
            if results.index((grid, time_ms)) < 3:
                print(line, end="")
        
        if len(results) > 3:
            print(f"| ... (see {output_file} for full list) ... |\n", end="")
            
        f.write(table_border)
        print(table_border, end="")
        
        best_grid = results[0][0]
        best_time = results[0][1]
        summary = f"Optimal Grid for {case_name} is {best_grid} ({best_time:.4f} ms).\n"
        f.write(summary)
        print(summary, end="")

print(f"\nAll results saved to {output_file}.")
