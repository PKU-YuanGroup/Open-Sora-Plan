from torch import nn

"""
We are currently working on the abstraction of the code structure, 
so there is no content here for the moment. 
It will be gradually optimized over time. 
Contributions from the open-source community are welcome.
"""
class VideoBaseAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def download_and_load_model(self):
        return None
    
    def encode(self, x, *args, **kwargs):
        pass

    def decode(self, encoding, *args, **kwargs):
        pass