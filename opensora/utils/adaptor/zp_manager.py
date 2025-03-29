import torch
import os
import torch.distributed as dist


class ZPManager(object):
    def __init__(self, zp_size=8):
        self.rank = int(os.getenv('RANK', '0'))
        self.world_size = int(os.getenv("WORLD_SIZE", '1'))
        self.zp_size = zp_size
        self.zp_group = None
        self.zp_rank = None
        self.is_initialized = False

    def init_group(self):
        if self.is_initialized:
            return

        self.is_initialized = True

        """Initialize the sequence parallel group."""
        num_zp_groups: int = self.world_size // self.zp_size
        for i in range(num_zp_groups):
            ranks = range(i * self.zp_size, (i + 1) * self.zp_size)
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.zp_group = group
                self.zp_rank = self.rank % self.zp_size


zp_manager = ZPManager()
