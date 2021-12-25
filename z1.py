import subprocess

result = subprocess.run(["python", "z2.py"],capture_output=True, text=True)

print("stdout:", result)
print("stderr:", result.stderr)