import numpy as np
import pymp
from time import perf_counter

"""
Run matrix multi
"""

# 3x3 matrix
A = [
	[12, 7, 3],
	[4, 5, 6],
	[7 ,8, 9]
]
# 3x4 matrix
B = [
	[5, 8, 1, 2],
	[6, 7, 3, 0],
	[4, 5, 9, 1]
]
# result is 3x4
result = [
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0]
]


# print(A)
# print(B)
# print()

"""
[114, 160, 60, 27]
[74, 97, 73, 14]
[119, 157, 112, 23]
"""

for k in range(5, 200, 5):
	print(k)
	row = k
	col = k
	A = np.random.rand(row, col)
	B = np.random.rand(row, col)
	# A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	# B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
	# A = pymp.shared.array((row, col))  # <--
	# B = pymp.shared.array((row, col))  # <--
	result = pymp.shared.array((row, col))

	# Option 1
	# -------
	result = pymp.shared.array((row, col))
	start_time = perf_counter()
	# iterating by row of A
	for i in range(len(A)):
		# iterating by coloum by B 
		for j in range(len(B[0])):
			# iterating by rows of B
			for k in range(len(B)):
				distances = np.linalg.norm(A, axis=1)
				y = np.argmin(distances)
				result[i][j] += A[i][k] * B[k][j]  
	""" for r in result:
		print(r) """
	# print(result[0][:5])
	end_time = perf_counter()
	print("Elapsed wall clock time = %g seconds." % (end_time-start_time))
	# -------

	# Option 2
	# -------
	""" result = pymp.shared.array((row, col))
	start_time = perf_counter()
	result = [[sum(a * b for a, b in zip(A_row, B_col)) 
							for B_col in zip(*B)]
									for A_row in A]

	"" " for r in result:
		print(r) "" "
	print(result[0][:5])
	end_time = perf_counter()
	print("Elapsed wall clock time = %g seconds.\n" % (end_time-start_time)) """
	# -------

	# Option 3
	# -------
	""" result = pymp.shared.array((row, col))
	start_time = perf_counter()
	result = np.dot(A,B)
	"" " for r in result:
		print(r) "" "
	print(result[0][:5])
	end_time = perf_counter()
	print("Elapsed wall clock time = %g seconds.\n" % (end_time-start_time)) """
	# -------

	# Option 4
	# -------
	result = pymp.shared.array((row, col))
	start_time = perf_counter()
	# iterating by row of A
	with pymp.Parallel(3) as p:
		# print(p.thread_num, A)
		# print(p.thread_num, B)
		# print()
		for i in range(len(A)):
			# iterating by coloum by B 
			for j in range(len(B[0])):
				# iterating by rows of B
				for k in p.range(len(B)):
					# p.print(p.thread_num, i, j)
					distances = np.linalg.norm(A, axis=1)
					y = np.argmin(distances)
					result[i][j] += A[i][k] * B[k][j]  
	""" for r in result:
		print(r) """
	# print(result[0][:5])
	end_time = perf_counter()
	print("Elapsed wall clock time = %g seconds." % (end_time-start_time))
	# -------

	print()
# python -m cProfile -s ncalls example.py > zzz.txt 