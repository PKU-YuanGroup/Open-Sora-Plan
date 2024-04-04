from .imagebase import imagebase_ae, imagebase_ae_stride, imagebase_ae_channel
from .videobase import videobase_ae, videobase_ae_stride, videobase_ae_channel

ae_stride_config = {}
ae_stride_config.update(imagebase_ae_stride)
ae_stride_config.update(videobase_ae_stride)

ae_channel_config = {}
ae_channel_config.update(imagebase_ae_channel)
ae_channel_config.update(videobase_ae_channel)

def getae(args):
    """deprecation"""
    ae = imagebase_ae.get(args.ae, None)
    if ae is None:
        ae = videobase_ae.get(args.ae, None)
    assert ae is not None
    return ae(args.ae)

def getae_wrapper(args):
    """deprecation"""
    ae = imagebase_ae.get(args.ae, None)
    if ae is None:
        ae = videobase_ae.get(args.ae, None)
    assert ae is not None
    return ae