
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


def logistic_map(x, rate):
    return rate * x * (1 - x)


def biffurication_map(x_0, R, N, M):
    rate = []
    population = []

    for r in R:
        x = x_0
        for i in range(0, N):
            x = logistic_map(x, r)
        for i in range(0, M):
            x = logistic_map(x, r)
            population.append(x)
            rate.append(r)
    return population, rate


def main():
    len_r = 5000
    N = 400
    M = 400
    r = np.linspace(0, 4, len_r)
    x_0 = np.random.rand()


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    r_chunk = np.array_split(r, size)[rank]

    if rank == 0:
      start_time = MPI.Wtime()

    population_chunk, rate_chunk = biffurication_map(x_0, r_chunk, N, M)

    if rank == 0:
      end_time = MPI.Wtime()
      t = end_time - start_time
      print("Size: ", size, "\tTime elapsed: ", np.round(t, 3))

    rate = comm.gather(rate_chunk, root=0)
    population = comm.gather(population_chunk, root=0)

    if rank == 0:
        rate = np.hstack(rate)
        population = np.hstack(population)


if __name__ == '__main__':
    main()
