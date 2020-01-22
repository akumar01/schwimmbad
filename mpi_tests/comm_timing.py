import numpy as np
import sys
from mpi4py import MPI
import schwimmbad 
import itertools
import time
import pdb

def dummy_task(task):
    time.sleep(5)
    return None

if __name__ == '__main__':
    # Test the time needed for communications (bcast, send, receive)

    config_str = sys.argv[1]
    print(config_str)
    comm = MPI.COMM_WORLD

    rank = comm.rank
    size = comm.size

    print('Comm size: %d' % size)

    if rank == 0:
        data = list(np.zeros((10000, 10000)))
    else:
        data = None

    t0 = time.time()
    data = comm.bcast(data)
    
    bcast_time = time.time() - t0

    print('Data broadcast time: %f' % bcast_time)
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

    # How long does it take for schwimmbad to do its stuff? Is this longer than the usual comm bcast operations?
    pool = schwimmbad.MPIPool(comm)

    task_dicts = [{'x' : np.random.random(size=(100,)), 'y' : np.random.random(size=(100,))} for i in range(2 * size)]

    pool.map(dummy_task, task_dicts)

    pool.close()
    

