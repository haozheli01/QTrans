"""QTrans parallel.mpi helpers."""

from mpi4py import MPI
from mpi4py.MPI import Intracomm
from qtrans.core.constants import QTcts
from typing import Any
from typing import List
from typing import Sequence
from typing import Union
import numpy as np
try:
    import resource
except ImportError:
    resource = None


def barrier_ranks(comm:Union[Intracomm, None]):
    if comm is not None:
        comm.Barrier()

def get_mpi_split_sections(data_len:int, rank_size:int):
    quotient, remainder = divmod(data_len, rank_size)
    counts = [quotient + 1 if i < remainder else quotient for i in range(rank_size)]
    split_indices = np.cumsum(counts[:-1])
    return split_indices

def split_into_rank(data:Sequence, axis:int, rank_id:int, rank_size:int, split_indices:Union[np.ndarray, Sequence[int]]=None, return_indices:bool=False):
    data:np.ndarray = np.asarray(data)
    if split_indices is None:
        data_len = data.shape[axis]
        split_indices = get_mpi_split_sections(data_len, rank_size)
    
    chunks = np.split(data, split_indices, axis=axis)
    local_chunk = chunks[rank_id]
    if return_indices:
        return local_chunk, split_indices
    else:
        return local_chunk


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Return the deterministic 64-bit SplitMix hash used by QTrans20260721."""
    values = np.asarray(values, dtype=np.uint64)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def get_rank_indices(data_len: int, rank_id: int, rank_size: int,
                     split_mode: str = "block") -> np.ndarray:
    """Return deterministic local indices for a one-dimensional data set."""
    if rank_size < 1:
        raise ValueError("rank_size must be positive.")
    if not (0 <= rank_id < rank_size):
        raise ValueError("rank_id is outside [0, rank_size).")
    if data_len < 0:
        raise ValueError("data_len must be non-negative.")
    mode = str(split_mode).strip().lower().replace('_', '-')
    indices = np.arange(data_len, dtype=int)
    if mode in ("block", "contiguous"):
        return np.array_split(indices, rank_size)[rank_id]
    if mode == "hash-cyclic":
        permutation = np.argsort(_splitmix64(indices), kind="stable")
        return indices[permutation][rank_id::rank_size]
    raise ValueError(f"Unsupported q_split_mode={split_mode!r}.")


def get_q_rank_indices(q_grid: np.ndarray, rank_id: int, rank_size: int,
                       split_mode: str = "hash-cyclic",
                       q_size: Union[np.ndarray, Sequence[int], None] = None) -> np.ndarray:
    """Map q points to ranks, matching QTrans20260721 hash-cyclic semantics."""
    q_grid = np.asarray(q_grid)
    if q_grid.ndim != 2 or q_grid.shape[1] != 3:
        raise ValueError("q_grid must have shape (nq, 3).")
    mode = str(split_mode).strip().lower().replace('_', '-')
    if mode in ("block", "contiguous"):
        return get_rank_indices(len(q_grid), rank_id, rank_size, mode)
    if mode != "hash-cyclic":
        raise ValueError(f"Unsupported q_split_mode={split_mode!r}.")
    if rank_size < 1:
        raise ValueError("rank_size must be positive.")
    if not (0 <= rank_id < rank_size):
        raise ValueError("rank_id is outside [0, rank_size).")
    if q_size is None:
        hash_input = np.arange(len(q_grid), dtype=np.uint64)
    else:
        q_size = np.asarray(q_size, dtype=np.int64)
        if q_size.shape != (3,) or np.any(q_size < 1):
            raise ValueError("q_size must contain three positive integers.")
        q_coords = np.rint(np.mod(q_grid, 1.0) * q_size[None, :]).astype(np.int64)
        q_coords = np.mod(q_coords, q_size[None, :]).astype(np.uint64)
        qs = q_size.astype(np.uint64)
        hash_input = q_coords[:, 0] + qs[0] * (q_coords[:, 1] + qs[1] * q_coords[:, 2])
    indices = np.arange(len(q_grid), dtype=int)
    permutation = np.argsort(_splitmix64(hash_input), kind="stable")
    return indices[permutation][rank_id::rank_size]


def split_by_rank_indices(data: Sequence, axis: int, indices: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= data.shape[axis]):
        raise ValueError("indices are outside the selected data axis.")
    return np.take(data, indices, axis=axis)

def gather_to_rank_root(data_local:Any, comm:Union[Intracomm, None], root:int=0) -> Union[List[Any], None]:
    data_type = type(data_local)
    if comm is None:
        data_all = [data_local]
    else:
        data_all = None
        if data_type == np.ndarray:
            data_all = gather_to_rank_root_array(data_local, comm, root)
        else:
            data_all = gather_to_rank_root_obj(data_local, comm, root)
    return data_all

def gather_to_rank_root_obj(data_local:Any, comm:Union[Intracomm, None], root:int=0) -> Union[List[Any], None]:
    """
    Gather data from all ranks to the root rank.

    Parameters:
    -----------
    data_local : Any
        Local data to be gathered from each rank.
    comm : Intracomm or None
        MPI communicator. If None, treated as serial mode.
    root : int (Default : 0)
        Rank ID to gather data to.

    Returns:
    --------
    data_gathered : list or None
        - On root rank: a list of gathered data from all ranks.
        - On other ranks: None.
        - In serial mode (comm is None): [data_local].
    """
    
    rank_size = comm.Get_size()
    rank = comm.Get_rank()

    if not (0 <= root < rank_size):
        raise ValueError(f"Invalid root rank {root}. Must be in range [0, {rank_size-1}]")

    if rank_size == 1:
        return [data_local]

    # Use point-to-point send/recv instead of comm.gather to avoid
    # the 2 GB pickle buffer overflow in mpi4py for large objects.
    if rank == root:
        data_all = [None] * rank_size
        data_all[root] = data_local
        for src in range(rank_size):
            if src == root:
                continue
            data_all[src] = comm.recv(source=src, tag=src)
    else:
        comm.send(data_local, dest=root, tag=rank)
        data_all = None
    return data_all

def gather_to_rank_root_array(data_local:np.ndarray, comm:Union[Intracomm, None], root:int=0) -> Union[List[np.ndarray], None]:
    """
    Gather data from all ranks to the root rank.

    Parameters:
    -----------
    data_local : Any
        Local data to be gathered from each rank.
    comm : Intracomm or None
        MPI communicator. If None, treated as serial mode.
    root : int (Default : 0)
        Rank ID to gather data to.

    Returns:
    --------
    data_gathered : list or None
        - On root rank: a list of gathered data from all ranks.
        - On other ranks: None.
        - In serial mode (comm is None): [data_local].
    """
    data_local:np.ndarray = np.asarray(data_local)
    
    rank_size = comm.Get_size()

    if not (0 <= root < rank_size):
        raise ValueError(f"Invalid root rank {root}. Must be in range [0, {rank_size-1}]")

    bytes_local = data_local.nbytes
    total_bytes = comm.allreduce(bytes_local, op=MPI.SUM)
    use_gatherv = total_bytes > QTcts.MAX_BUFFER_BYTES

    if use_gatherv:
        data_all = gather_to_rank_root_large_array(data_local, comm, root)
    else:
        data_all = comm.gather(data_local, root=root)
    return data_all

def reduce_to_rank_root(data_local: Any, 
                        comm: Union[Intracomm, None], 
                        operation: str = "sum", 
                        root: int = 0) -> Any:
    """
    Reduces `data_local` from all ranks to the `root` rank using MPI reduction.

    Parameters:
        data_local : Any
            Local data on each process (should be NumPy array or scalar).
        comm : mpi4py.MPI.Intracomm or None
            The MPI communicator. If None, returns data_local.
        operation : str
            Reduction operation ('sum', 'prod', 'max', 'min').
        root : int
            The rank that will receive the reduced result.

    Returns:
        Reduced data on root rank; None on other ranks.
    """
    if comm is None:
        return data_local

    rank = comm.Get_rank()
    size = comm.Get_size()

    if not (0 <= root < size):
        raise ValueError(f"Invalid root rank {root}. Must be in range [0, {size-1}]")

    op_map = {
        'sum': MPI.SUM, '+': MPI.SUM,
        'prod': MPI.PROD, '*': MPI.PROD,
        'max': MPI.MAX,
        'min': MPI.MIN,
    }

    op_key = operation.lower()
    if op_key not in op_map:
        raise NotImplementedError(f"The MPI reduce operation '{operation}' is not supported.")

    mpi_op = op_map[op_key]

    # Ensure data is a NumPy array
    if not isinstance(data_local, np.ndarray):
        data_local = np.array(data_local)

    # Allocate output buffer on root only
    data_all = np.empty_like(data_local) if rank == root else None

    comm.Reduce(data_local, data_all, op=mpi_op, root=root)

    return data_all

def get_mpi_data_type(arr:np.ndarray):
    if arr.dtype == np.float64:
        return MPI.DOUBLE
    elif arr.dtype == np.float32:
        return MPI.FLOAT
    elif arr.dtype == np.int32:
        return MPI.INT
    elif arr.dtype == np.int64:
        return MPI.LONG
    elif arr.dtype == np.complex64:
        return MPI.COMPLEX
    elif arr.dtype == np.complex128:
        return MPI.DOUBLE_COMPLEX
    else:
        raise ValueError(f"Unsupported numpy dtype {arr.dtype}")

def gather_to_rank_root_large_array(data_local:Any, comm:Union[Intracomm, None], root:int=0) -> Union[List[Any], None]:
    
    rank_size = comm.Get_size()

    if not (0 <= root < rank_size):
        raise ValueError(f"Invalid root rank {root}. Must be in range [0, {rank_size-1}]")

    rank = comm.Get_rank()
    sendbuf = data_local.flatten()
    sendcount = sendbuf.size
    recvcounts = comm.gather(sendcount, root=0)
    shape_local = data_local.shape
    shape_all = comm.gather(shape_local, root=0)

    if rank == root:
        displs = np.insert(np.cumsum(recvcounts[:-1]), 0, 0)
        recvbuf = np.empty(sum(recvcounts), dtype=sendbuf.dtype)
    else:
        recvbuf = None
        displs = None
    data_type = get_mpi_data_type(data_local)
    comm.Gatherv(sendbuf, [recvbuf, recvcounts, displs, data_type], root=root)
    if rank == root:
        data_all = []
        offset = 0
        for i, shape in enumerate(shape_all):
            size = np.prod(shape)
            arr_i = recvbuf[offset : offset+size].reshape(shape)
            data_all.append(arr_i)
            offset += size
    else:
        data_all = None
    return data_all

def print_peak_memeory(comm:Union[Intracomm, None], rank_id:int):
    local_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    if comm is not None:
        total_rss_gb = comm.reduce(local_rss_gb, op=MPI.SUM, root=0)
    else:
        total_rss_gb = local_rss_gb
    if rank_id == 0:
        print(f"Total peak memory across all ranks: {total_rss_gb:.2f} GB")
