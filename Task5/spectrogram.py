
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def create_wave(t):
    y = 13 * np.sin(1.5 * t) * np.exp(-(t - 6 * 3.5 * np.pi)**2/2/20**2)
    y += 7 * np.sin(3.3 * t) * np.exp(-(t - 4 * 3* np.pi)**2/2/20**2)
    y += 5 * np.sin(4.13 * t) * np.exp(-(t - 13 * np.pi)**2/2/20**2)
    y += 17 * np.sin(t) * np.exp(-(t - np.pi)**2/2/20**2)
    return y

def get_specgram(t, y, start_pos, stop_pos, nwindowsteps=10000):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    window_positions = np.linspace(start_pos, stop_pos, nwindowsteps)

    start_time = MPI.Wtime()
    wp_chunk = np.array_split(window_positions, size)[rank]

    window_width = 2.0*2*np.pi

    specgram_chunk = np.empty([len(t), len(wp_chunk)])

    for i, window_position in enumerate(wp_chunk):
        y_window = y * np.exp(- (t - window_position) ** 2 / 2 / window_width ** 2)
        specgram_chunk[:, i] = np.abs(np.fft.fft(y_window))

    specgram = comm.gather(specgram_chunk, root=0)

    if rank == 0:
        cn = np.concatenate(specgram, axis=1)
        end_time = MPI.Wtime()
        t = end_time - start_time
        print("Size: ", size, "\tTime elapsed: ", np.round(t, 3))
        return cn

def main():
    t = np.linspace(-20 * 2 * np.pi, 20 * 2 * np.pi, 2**12)
    y = create_wave(t)
    spectogram = get_specgram(t, y, -20 * 2 * np.pi, 20 * 2 * np.pi, nwindowsteps=10000)

if __name__ == '__main__':
    main()
