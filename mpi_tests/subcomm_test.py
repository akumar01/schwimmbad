from schwimmbad import MPIPool
from mpi4py import MPI
import numpy as np
import time


def dummy_task(task):
	
	stuff, subcomm = task
	print('Rank %d playing a role in task!' % subcomm.rank)
	return 0


if __name__ == '__main__':
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	subgroups = np.array_split(np.arange(size), 4)
	pool = MPIPool.comm(comm, subgroups=subgroups)

	tasks = np.arange(100)
	pool.map(dummy_task, tasks)
	pool.close()