import numpy as np
from mpi4py import MPI
import schwimmbad 
import itertools
import time


def dummy_task(task):
	return None

if __name__ == '__main__':
	# Test the time needed for communications (bcast, send, receive)

	comm = MPI.COMM_WORLD

	rank = comm.rank
	size = comm.size


	if rank == 0:
		data = list(np.zeros((10000, 10000)))
	else:
		data = None

	t0 = time.time()
	data = comm.bcast(data)
	print('Data broadcast time: %f' % (time.time() - t0))
	comm.barrier()

	# Time sends from root to all other ranks
	if rank == 0:
		data2 = list(np.zeros((1000, 1000)))
		for i in range(1, size):
			t0 = time.time()
			comm.send(data2, dest=i, tag=i)
			print('Data send to rank %d time: %f' % (i, time.time() - t0))
	else:
		t0 = time.time()
		data = comm.recv(source=0, tag=rank)
		print('Rank %d took %f s to receive' % (rank, time.time() - t0))

	comm.barrier()

	# Random pairwise ranks (non root)
	combos = np.itertools.combinations(np.arange(size), 2)
	rankpairs = np.random.choice(combos, 50)

	for rank_pair in rank_pairs:

		if rank == rank_pair[0]:
			data3 = np.zeros((1000, 1000))
			t0 = time.time()
			comm.send(data2, dest=rank_pair[1], tag=rank_pair[1])
			print('Data send from rank %d to rank %d time: %f' % (rank_pair[0], rank_pair[1], time.time() - t0))
		elif rank == rank_pair[1]:
			t0 = time.time()
			data = comm.recv(source=0, tag=rank)
			print('Rank %d took from rank %d in %f s ' % (rank_pair[1], rank_pair[0], time.time() - t0))
		else:
			continue

	comm.barrier()

	# Arrange subcommunicators and test their internal communication times
	numproc = comm.Get_size()

	comm_splits = 4

	# Use array split to do comm.split

	# Take the root node and set it aside - this is what schwimmbad will use to coordinate
	# the other groups

	ranks = np.arange(numproc)
	split_ranks = np.array_split(ranks, comm_splits)
	# if rank == 0:
	#     color = 0
	# else:
	color = [i for i in np.arange(comm_splits) if rank in split_ranks[i]][0]
	subcomm_roots = [split_ranks[i][0] for i in np.arange(comm_splits)]
	subcomm = comm.Split(color, rank)

	nchunks = comm_splits
	subrank = subcomm.rank
	numproc = subcomm.Get_size()

	# Create a group including the root of each subcomm (unused at the moment)
	global_group = comm.Get_group()
	root_group = MPI.Group.Incl(global_group, subcomm_roots)
	root_comm = comm.Create(root_group)

	# Broadcast data from root of each subcommunicator
	if subcomm.rank == 0:
		data = list(np.zeros((10000, 10000)))
	else:
		data = None

	t0 = time.time()
	data = subcomm.bcast(data)
	print('Color %d subcomm data broadcast time: %f' % (color, time.time() - t0))
	comm.barrier()

	# Gather data from root of each subcommunicator
	t0 = time.time()
	data = root_comm.gather(data)
	print('Root group gather time: %f' (time.time() - t0))


	comm.barrier()

	# How long does it take for schwimmbad to do its stuff? Is this longer than the usual comm bcast operations?
	pool = schwimmbad.MPIPool(comm)

	task_dicts = [{'x' : np.random.random(size=(100,)), 'y' : np.random.random(size=(100,))} for i in range(10000)]

	with pool.map(task_dicts, dummy_task) as results:
		pass

	pool.close()
	

