import os
import subprocess
import json
import tempfile
import sys

# Candidate grids to test for M=256
# T = 256 for M=256 (assuming glm5 config: N=256, K=6144, TOPK=8)
# We test T itself, SMS, and various multiples up to SMS * BMAX (132 * 6 = 792)
candidates = [
    8, 16, 32, 64, 128, 132, 192, 256, 264, 384, 396, 512, 528, 640, 660, 768, 792
]
# Add some fine-grained candidates around the theoretical optimal and boundaries
candidates += list(range(200, 300, 12))
candidates = sorted(list(set(candidates)))

print("Step 1: Recompiling extension with MARLIN_MOE_FORCE_GRID support...")
# build_cmd = f"PYTHONPATH={os.getcwd()}/python {sys.executable} setup.py build_ext --inplace"
# try:
#     subprocess.run(build_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# except subprocess.CalledProcessError:
#     print("Warning: Build failed. Assuming it's already built or running in a CPU-only env for testing.")
print(f"\nStep 2: Benchmarking {len(candidates)} candidate grids for M=256...")

results = []

for grid in candidates:
    env = os.environ.copy()
    env["MARLIN_MOE_FORCE_GRID"] = str(grid)
    env["PYTHONPATH"] = f"{os.getcwd()}/python"
    
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
                    import re
                    match = re.search(r'([0-9]+\.[0-9]+)\s*ms', line)
                    if match:
                        time_ms = float(match.group(1))
                        break
                        
        if time_ms is not None:
            results.append((grid, time_ms))
            print(f"  Grid {grid:3d} -> {time_ms:.4f} ms")
        else:
            print(f"  Grid {grid:3d} -> Failed to parse time. (Check if GPU/torch is available)")
            # Print a snippet of the error to help debugging
            error_snippet = "\n".join(proc.stdout.split("\n")[-5:])
            print(f"    Snippet: {error_snippet}")
            
    except Exception as e:
        print(f"  Grid {grid:3d} -> Error: {e}")
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

print("\nStep 3: Results Summary")
if not results:
    print("No valid results collected. Please check the benchmark script output.")
    sys.exit(1)

# Sort by execution time (ascending)
results.sort(key=lambda x: x[1])

print("+" + "-"*30 + "+")
print("| Grid Size | Execution Time (ms) |")
print("+" + "-"*30 + "+")
for grid, time_ms in results:
    if grid == results[0][0]:
        print(f"| {grid:9d} | {time_ms:17.4f} * |  <-- BEST")
    else:
        print(f"| {grid:9d} | {time_ms:17.4f}   |")
print("+" + "-"*30 + "+")

best_grid = results[0][0]
best_time = results[0][1]
print(f"\nOptimal Grid for M=256 is {best_grid} ({best_time:.4f} ms).")
print("You can record this in your lookup table.")

