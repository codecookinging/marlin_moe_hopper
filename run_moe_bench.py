import subprocess
print("Checking benchmark script...")
subprocess.run(["python3", "tests/test_marlin_moe.py"], capture_output=False)
