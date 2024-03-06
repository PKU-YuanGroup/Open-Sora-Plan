from .imagebase import imagebase_ae, vae, vqvae
from .videobase import videobase_ae, videovae, videovqvae
from .videobase import (
    VideoGPTConfiguration,
    VideoGPTVQVAE
)

def getae(args):
    """deprecation"""
    ae = imagebase_ae.get(args.ae, None)
    if ae is None:
        ae = videobase_ae.get(args.ae, None)
    assert ae is not None
    return ae(args.ae)