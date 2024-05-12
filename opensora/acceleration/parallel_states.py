import torch
import torch_npu
import torch.distributed as dist
import os
try:
    from lcalib.functional import lcal_initialize
    enable_LCCL = True
except:
    lcal_initialize = None
    enable_LCCL = False

class LCCL_INFO:
    def __init__(self):
        self._COMM_WORLD = None
        self.world_size = 0
        self.rank = -1
class HCCL_INFO:
    def __init__(self):
        self.group = None
        self.world_size = 0
        self.rank = -1

lccl_info = LCCL_INFO()
hccl_info = HCCL_INFO()

def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        _SEQUENCE_PARALLEL_STATE = False

def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state

def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE

def initialize_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    assert world_size % sequence_parallel_size == 0, "world_size must be divisible by sequence_parallel_size"
    # hccl
    hccl_info.world_size = sequence_parallel_size
    hccl_info.rank = rank
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            hccl_info.group = group

    if enable_LCCL:
        assert sequence_parallel_size == 8, "sequence_parallel_size should be 8 when enable_LCCL is True"
        rank %= sequence_parallel_size
        lccl_info.world_size = sequence_parallel_size
        lccl_info._COMM_WORLD = lcal_initialize(rank, sequence_parallel_size)
        lccl_info.rank = rank

def destroy_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    if not enable_LCCL:
        dist.destroy_process_group()
