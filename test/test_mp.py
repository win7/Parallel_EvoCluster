from __future__ import print_function
from time import perf_counter

import pymp  # <--
import numpy as np

length = 3
# array = np.zeros(length, dtype="float")
array = pymp.shared.array(length, dtype="float")
array.fill(float("inf"))
print(array)
with pymp.Parallel(length) as p:
    for index in p.range(length):
        with p.lock:
            array[p.thread_num] = 10 + p.thread_num
            print(p.thread_num, array)
print(array)

""" length = 100
ex_array = np.zeros(length, dtype='uint8')
for index in range(0, length):
    ex_array[index] = 1
    print('Yay! {} done!'.format(index))

print("---")

ex_array = pymp.shared.array(length, dtype='uint8')
print(ex_array)

# ex_array = np.zeros(length, dtype='uint8')
with pymp.Parallel(4) as p:
    for index in p.range(length):
        ex_array[index] = 1
        # The parallel print function takes care of asynchronous output.
        p.print('Yay! {} done!'.format(index))
print(ex_array) """

""" nx = 1201
ny = 1201
 
# Solution and previous solution arrays
# sol = pymp.shared.array((nx,ny))  # <--
# soln = pymp.shared.array((nx,ny))  # <--
sol = numpy.zeros((nx,ny))
soln = sol.copy()

 
for j in range(0,ny-1):
    sol[0,j] = 10.0
    sol[nx-1,j] = 1.0
# end for
 
for i in range(0,nx-1):
    sol[i,0] = 0.0
    sol[i,ny-1] = 0.0
# end for
 
# Iterate
start_time = perf_counter()
with pymp.Parallel(4) as p:  # <--
    for kloop in range(1,100):
        soln = sol.copy()
 
        for i in p.range(1,nx-1):
            for j in p.range (1,ny-1):
                sol[i,j] = 0.25 * (soln[i,j-1] + soln[i,j+1] + soln[i-1,j] + soln[i+1,j])
            # end j for loop
        # end i for loop
    # end kloop for loop
# end with
end_time = perf_counter()
 
print(' ')
print(sol[0][:20])
print('Elapsed wall clock time = %g seconds.' % (end_time-start_time) )
print(' ') """

"""
# Serial
import numpy
from time import perf_counter
 
nx = 1201
ny = 1201
 
# Solution and previous solution arrays
sol = numpy.zeros((nx,ny))
soln = sol.copy()
  
for j in range(0,ny-1):
    sol[0,j] = 10.0
    sol[nx-1,j] = 1.0
# end for
  
for i in range(0,nx-1):
    sol[i,0] = 0.0
    sol[i,ny-1] = 0.0
# end for
  
# Iterate
start_time = perf_counter()
for kloop in range(1,100):
    soln = sol.copy()
 
    for i in range(1,nx-1):
        for j in range (1,ny-1):
            sol[i,j] = 0.25 * (soln[i,j-1] + soln[i,j+1] + soln[i-1,j] + soln[i+1,j])
        # end j for loop
    # end i for loop
#end for
end_time = perf_counter()
 
print(' ')
print(sol[0][:20])
print('Elapsed wall clock time = %g seconds.' % (end_time-start_time) )
print(' ')
"""