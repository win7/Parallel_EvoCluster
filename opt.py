import os

print(123)
def run():
    result = os.system("mpirun -np 24 --oversubscribe python selector.py")
