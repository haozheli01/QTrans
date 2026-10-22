"""QTrans parallel.mpi helpers."""

from mpi4py import MPI
from mpi4py.MPI import Intracomm
from qtrans.core.constants import QTcts
from typing import Any
from typing import List
from typing import Sequence
from typing import Union
import numpy as np
import pickle
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
        # Assign ownership directly from the deterministic hash.  The old
        # implementation sorted all hashes and then strided the permutation;
        # that made every rank allocate O(N) permutation/index arrays.  The
        # owner map preserves deterministic coverage while using one boolean
        # mask and local indices.
        owners = _splitmix64(indices.astype(np.uint64)) % np.uint64(rank_size)
        return indices[owners == np.uint64(rank_id)]
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
    owners = _splitmix64(hash_input) % np.uint64(rank_size)
    return indices[owners == np.uint64(rank_id)]


def get_q_rank_index_parts(q_grid: np.ndarray, rank_size: int,
                           split_mode: str = 'contiguous',
                           q_size: Union[np.ndarray, Sequence[int], None] = None):
    """Build all q owner parts once, for a root-only q-grid partition.

    The caller should invoke this on the root rank and scatter the returned
    coordinate/index parts.  In hash-cyclic mode the hash and owner mask are
    computed once instead of once per rank.
    """
    q_grid = np.asarray(q_grid)
    if q_grid.ndim != 2 or q_grid.shape[1] != 3:
        raise ValueError('q_grid must have shape (nq, 3).')
    if rank_size < 1:
        raise ValueError('rank_size must be positive.')
    mode = str(split_mode).strip().lower().replace('_', '-')
    indices = np.arange(len(q_grid), dtype=int)
    if mode in ('block', 'contiguous'):
        return list(np.array_split(indices, rank_size))
    if mode != 'hash-cyclic':
        raise ValueError(f'Unsupported q_split_mode={split_mode!r}.')
    if q_size is None:
        hash_input = np.arange(len(q_grid), dtype=np.uint64)
    else:
        q_size = np.asarray(q_size, dtype=np.int64)
        if q_size.shape != (3,) or np.any(q_size < 1):
            raise ValueError('q_size must contain three positive integers.')
        q_coords = np.rint(np.mod(q_grid, 1.0) * q_size[None, :]).astype(np.int64)
        q_coords = np.mod(q_coords, q_size[None, :]).astype(np.uint64)
        qs = q_size.astype(np.uint64)
        hash_input = q_coords[:, 0] + qs[0] * (
            q_coords[:, 1] + qs[1] * q_coords[:, 2])
    owners = _splitmix64(hash_input) % np.uint64(rank_size)
    return [indices[owners == np.uint64(rank)] for rank in range(rank_size)]


def split_by_rank_indices(data: Sequence, axis: int, indices: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= data.shape[axis]):
        raise ValueError("indices are outside the selected data axis.")
    return np.take(data, indices, axis=axis)


def iter_bcast_array_by_rank_chunks(data_local, comm, chunk_bytes=512 * 1024 ** 2):
    """Yield ``(owner, row_offset, chunk)`` in rank order using typed Bcast.

    A consumer can process each owner's wavefunctions without retaining the
    arrays belonging to every rank. Every MPI payload is strictly bounded.
    """
    data_local = np.ascontiguousarray(data_local)
    if comm is None:
        yield 0, 0, data_local.copy()
        return
    rank = comm.Get_rank()
    shape_tail = tuple(data_local.shape[1:])
    dtype = data_local.dtype
    metadata = comm.allgather((int(data_local.shape[0]), shape_tail, dtype.str))
    if any(tuple(item[1]) != shape_tail or item[2] != dtype.str for item in metadata):
        raise RuntimeError("Inconsistent array shape or dtype across ranks.")
    row_bytes = max(1, int(np.prod(shape_tail, dtype=np.int64)) * dtype.itemsize)
    if chunk_bytes < dtype.itemsize:
        raise ValueError("MPI chunk limit is smaller than one array element.")
    rows_per_chunk = max(1, int(chunk_bytes) // row_bytes)
    for owner, (count, _, _) in enumerate(metadata):
        for offset in range(0, count, rows_per_chunk):
            nrows = min(rows_per_chunk, count - offset)
            if rank == owner:
                chunk = np.ascontiguousarray(data_local[offset:offset + nrows])
            else:
                chunk = np.empty((nrows,) + shape_tail, dtype=dtype)
            if chunk.nbytes <= chunk_bytes:
                comm.Bcast(chunk, root=owner)
            else:
                flat = chunk.reshape(-1)
                nelem = max(1, int(chunk_bytes) // dtype.itemsize)
                for start in range(0, flat.size, nelem):
                    comm.Bcast(flat[start:start + nelem], root=owner)
            yield owner, offset, chunk


def bcast_array_by_rank_chunks(data_local, comm, chunk_bytes=512 * 1024 ** 2):
    """Collect typed rank broadcasts in rank order (legacy superconductor API)."""
    data_local = np.ascontiguousarray(data_local)
    if comm is None:
        return data_local.copy()
    counts = comm.allgather(int(data_local.shape[0]))
    starts = np.insert(np.cumsum(counts[:-1], dtype=np.int64), 0, 0)
    result = np.empty((sum(counts),) + data_local.shape[1:], dtype=data_local.dtype)
    for owner, offset, chunk in iter_bcast_array_by_rank_chunks(
            data_local, comm, chunk_bytes):
        result[starts[owner] + offset:starts[owner] + offset + len(chunk)] = chunk
    return result


def allreduce_array_in_place_chunks(data, comm, chunk_bytes=512 * 1024 ** 2):
    """Sum a contiguous ndarray, chunking only when its payload is too large."""
    if comm is None:
        return
    if not data.flags.c_contiguous:
        raise ValueError("Chunked Allreduce requires a C-contiguous array.")
    flat = data.reshape(-1)
    nelem = max(1, int(chunk_bytes) // data.dtype.itemsize)
    for start in range(0, flat.size, nelem):
        comm.Allreduce(MPI.IN_PLACE, flat[start:start + nelem], op=MPI.SUM)


def gather_array_to_root_chunks(data_local, comm, root=0,
                                chunk_bytes=512 * 1024 ** 2):
    """Gather trailing-k-axis arrays with one typed Gatherv when small.

    If the combined receive payload exceeds the bound, transfer each sender
    in bounded typed chunks to avoid a 32-bit MPI count or pickle limit.
    """
    data_local = np.ascontiguousarray(data_local)
    if comm is None:
        return [data_local.copy()]
    rank = comm.Get_rank()
    metadata = comm.allgather((data_local.shape, data_local.dtype.str))
    if any(item[0][:-1] != data_local.shape[:-1] or
           item[1] != data_local.dtype.str for item in metadata):
        raise RuntimeError("Inconsistent gather array shape or dtype.")
    counts = [int(np.prod(shape, dtype=np.int64)) for shape, _ in metadata]
    if (sum(counts) * data_local.dtype.itemsize <= chunk_bytes and
            sum(counts) < np.iinfo(np.int32).max):
        displacements = np.insert(np.cumsum(counts[:-1], dtype=np.int64), 0, 0)
        received = (np.empty(sum(counts), dtype=data_local.dtype)
                    if rank == root else None)
        comm.Gatherv(data_local.reshape(-1),
                     [received, counts, displacements,
                      get_mpi_data_type(data_local)] if rank == root else None,
                     root=root)
        if rank == root:
            return [received[displacements[i]:displacements[i] + count]
                    .reshape(metadata[i][0]) for i, count in enumerate(counts)]
        return None
    nelem = max(1, int(chunk_bytes) // data_local.dtype.itemsize)
    if rank == root:
        result = [None] * comm.Get_size()
        result[root] = data_local
        for owner, (shape, _) in enumerate(metadata):
            if owner == root:
                continue
            received = np.empty(shape, dtype=data_local.dtype)
            flat = received.reshape(-1)
            for start in range(0, flat.size, nelem):
                comm.Recv(flat[start:start + nelem], source=owner, tag=941)
            result[owner] = received
        return result
    flat = data_local.reshape(-1)
    for start in range(0, flat.size, nelem):
        comm.Send(flat[start:start + nelem], dest=root, tag=941)
    return None


def bcast_array_from_root_chunks(data_root, comm, root=0,
                                 chunk_bytes=512 * 1024 ** 2):
    """Broadcast one contiguous ndarray with bounded typed payloads."""
    if comm is None:
        return np.ascontiguousarray(data_root).copy()
    rank = comm.Get_rank()
    if rank == root:
        data = np.ascontiguousarray(data_root)
        metadata = (data.shape, data.dtype.str)
    else:
        metadata = None
    shape, dtype_str = comm.bcast(metadata, root=root)
    if rank != root:
        data = np.empty(shape, dtype=np.dtype(dtype_str))
    flat = data.reshape(-1)
    nelem = max(1, int(chunk_bytes) // data.dtype.itemsize)
    for start in range(0, flat.size, nelem):
        comm.Bcast(flat[start:start + nelem], root=root)
    return data

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

_OBJECT_CHUNK_BYTES = 512 * 1024 ** 2


def _send_pickled_object_chunks(data: Any, comm: Intracomm, dest: int,
                                control_tag: int = 9100,
                                payload_tag: int = 9101) -> None:
    """Send one Python object without a 2-GiB mpi4py pickle message.

    ``comm.send`` pickles the complete object into one MPI message.  Graph
    packing can produce a single rank-local dictionary larger than the
    32-bit-count limit, so serialize once and transfer the resulting bytes in
    bounded typed ``MPI.BYTE`` messages.  The receiver gets the byte count in
    a small control message and reconstructs the original object exactly.
    """
    payload = pickle.dumps(data, protocol=5)
    payload_view = np.frombuffer(payload, dtype=np.uint8)
    comm.send(int(payload_view.size), dest=dest, tag=control_tag)
    for start in range(0, payload_view.size, _OBJECT_CHUNK_BYTES):
        comm.Send(payload_view[start:start + _OBJECT_CHUNK_BYTES],
                  dest=dest, tag=payload_tag)


def _recv_pickled_object_chunks(comm: Intracomm, source: int,
                                control_tag: int = 9100,
                                payload_tag: int = 9101) -> Any:
    """Receive an object sent by ``_send_pickled_object_chunks``."""
    payload_size = int(comm.recv(source=source, tag=control_tag))
    if payload_size < 0:
        raise RuntimeError(f"Invalid pickled object size from rank {source}: {payload_size}")
    payload_view = np.empty(payload_size, dtype=np.uint8)
    for start in range(0, payload_size, _OBJECT_CHUNK_BYTES):
        comm.Recv(payload_view[start:start + _OBJECT_CHUNK_BYTES],
                  source=source, tag=payload_tag)
    return pickle.loads(payload_view)


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

    # Use bounded point-to-point typed transfers instead of comm.gather or
    # comm.send, both of which can create one >2-GiB pickle MPI message.
    if rank == root:
        data_all = [None] * rank_size
        data_all[root] = data_local
        for src in range(rank_size):
            if src == root:
                continue
            data_all[src] = _recv_pickled_object_chunks(comm, source=src)
    else:
        _send_pickled_object_chunks(data_local, comm, dest=root)
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
