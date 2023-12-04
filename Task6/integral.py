
import numpy as np
from mpi4py import MPI


def f(x):
    return (x * x - x + 2) / (x ** 4 - 5 * x * x + 4)


def solver_parallel(a, b, n):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = MPI.Wtime()

    n_ = n // size
    a_ = a + rank * n_ * (b - a) / n
    b_ = a_ + n_ * (b - a) / n

    x_ = np.linspace(a_, b_, n_)

    f_sum = 0
    for i in range(1, n_ - 1):
        f_sum += f(x_[i])
    f_sum += (f(a_) + f(b_)) / 2

    f_sum = comm.reduce(f_sum, op=MPI.SUM, root=0)

    if rank == 0:
        h = (b - a) / (n - 1)
        result = f_sum * h
        end_time = MPI.Wtime()
        t = end_time - start_time 
        print("Size: ", size, "\tTime elapsed: ", np.round(t, 4))


a = 7.0
b = 9.0
n = 100000

solver_parallel(a, b, n)
