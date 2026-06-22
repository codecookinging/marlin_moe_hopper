import os
import subprocess
import json
import tempfile
import sys
import re
from concurrent.futures import ThreadPoolExecutor

# Candidate grids to test for M=256
# T = 256 for M=256 (assuming glm5 config: N=256, K=6144, TOPK=8)
# We test T itself, SMS, and various multiples up to SMS * BMAX (132 * 6 = 792)
# Add some fine-grained candidates around the theoretical optimal and boundaries
candidates = list(range(8, 792, 8))
candidates = sorted(list(set(candidates)))

print("Step 1: Skipping recompilation (assuming already built)...")
# build_cmd = f"PYTHONPATH={os.getcwd()}/python {sys.executable} setup.py build_ext --inplace"
# try:
#     subprocess.run(build_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# except subprocess.CalledProcessError:
#     print("Warning: Build failed. Assuming it's already built or running in a CPU-only env for testing.")

print(f"\nStep 2: Benchmarking {len(candidates)} candidate grids for M=256 across 8 GPUs...")

def run_chunk(gpu_id, chunk):
    results = []
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
            "--cases", "glm5_prefill_m256",
            "--output-json", tmp_json
        ]
        
        try:
            # Run the benchmark
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Read the JSON output
            time_ms = None
            if os.path.exists(tmp_json) and os.path.getsize(tmp_json) > 0:
                with open(tmp_json, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        record = data[0]
                        for key in ['time_ms', 'mean_ms', 'batch_ms', 'p50_ms', 'latency_ms', 'latency']:
                            if key in record:
                                time_ms = float(record[key])
                                break
            
            # Fallback to parsing stdout if JSON parsing failed
            if time_ms is None:
                for line in proc.stdout.split('\n'):
                    if 'glm5_prefill_m256' in line and 'ms' in line:
                        match = re.search(r'([0-9]+\.[0-9]+)\s*ms', line)
                        if match:
                            time_ms = float(match.group(1))
                            break
                            
            if time_ms is not None:
                results.append((grid, time_ms))
                print(f"[GPU {gpu_id}] Grid {grid:3d} -> {time_ms:.4f} ms")
            else:
                print(f"[GPU {gpu_id}] Grid {grid:3d} -> Failed to parse time.")
                
        except Exception as e:
            print(f"[GPU {gpu_id}] Grid {grid:3d} -> Error: {e}")
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
    return results

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

# Sort by execution time (ascending)
all_results.sort(key=lambda x: x[1])

output_file = "grid_search_results.txt"
with open(output_file, "w") as f:
    f.write("+" + "-"*30 + "+\n")
    f.write("| Grid Size | Execution Time (ms) |\n")
    f.write("+" + "-"*30 + "+\n")
    for grid, time_ms in all_results:
        if grid == all_results[0][0]:
            line = f"| {grid:9d} | {time_ms:17.4f} * |  <-- BEST\n"
        else:
            line = f"| {grid:9d} | {time_ms:17.4f}   |\n"
        f.write(line)
        print(line, end="")
    f.write("+" + "-"*30 + "+\n")

best_grid = all_results[0][0]
best_time = all_results[0][1]
summary = f"\nOptimal Grid for M=256 is {best_grid} ({best_time:.4f} ms).\nResults saved to {output_file}."
print(summary)
with open(output_file, "a") as f:
    f.write(summary + "\n")

