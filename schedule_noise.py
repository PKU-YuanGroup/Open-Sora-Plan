from diffusers.schedulers import DDPMScheduler, DDIMScheduler, CogVideoXDDIMScheduler
import matplotlib.pyplot as plt
import imageio
import torch
import decord
from opensora.models.causalvideovae import ae_stride_config, ae_wrapper
from torchvision import transforms
from torchvision.transforms import Lambda
from opensora.dataset.transform import ToTensorVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, NormalizeVideo, ToTensorAfterResize
from opensora.utils.sample_utils import save_video_grid
from tqdm import tqdm

def get_scheduler(name, beta_start, beta_end, beta_schedule, zero_snr=True, snr_shift_scale=3):
    kwargs = dict(
        beta_start=beta_start, beta_end=beta_end,
        beta_schedule=beta_schedule, rescale_betas_zero_snr=zero_snr
    )
    if name == 'ddpm':
        scheduler_cls = DDPMScheduler
    elif name == 'ddim':
        scheduler_cls = DDIMScheduler
    elif name == 'cogvideox':
        scheduler_cls = CogVideoXDDIMScheduler
        kwargs['snr_shift_scale'] = snr_shift_scale
    scheduler = scheduler_cls(**kwargs)
    return scheduler

def get_transform(args):
    transform = transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo((args.max_height, args.max_width)), 
            Lambda(lambda x: 2. * x - 1.)
        ])  # also work for img, because img is video when frame=1
    return transform

def get_video(path, num_frames, transform, device, weight_dtype):
    decord_vr = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
    fps = decord_vr.get_avg_fps() 
    total_frames = num_frames or len(decord_vr)
    video_data = decord_vr.get_batch(list(range(total_frames))).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
    video = transform(video_data)  # T C H W -> T C H W
    video = video.transpose(0, 1)  # T C H W -> C T H W
    video = video.unsqueeze(0).to(device, dtype=weight_dtype)
    return video, fps

def add_noise(model_input, noise, noise_scheduler, timesteps):
    latents = noise_scheduler.add_noise(model_input, noise, timesteps)
    return latents

args = type('args', (), 
    {
        'ae': 'WFVAEModel_D8_4x8x8', 
        'ae_path': "/storage/lcm/Causal-Video-VAE/results/WFVAE_DISTILL_FORMAL", 
        'max_height': 1024,
        'max_width': 1024,
        'num_frames': 45,
        'enable_tiling': True, 
    }
    )

weight_dtype = torch.bfloat16
device = torch.device('cuda:0')
vae = ae_wrapper[args.ae](args.ae_path)
vae.vae = vae.vae.to(device=device, dtype=weight_dtype).eval()
if args.enable_tiling:
    vae.vae.enable_tiling()

# scheduler_1_2 = get_scheduler('ddpm', 0.0001, 0.02, "linear", True)
# scheduler_1_2_sl = get_scheduler('ddpm', 0.0001, 0.02, "scaled_linear", True)
# scheduler_beta_sl = get_scheduler('ddpm', 0.00085, 0.012, "scaled_linear", True)
# scheduler_cogvideox = get_scheduler('cogvideox', 0.00085, 0.012, "scaled_linear", True)
# schedulers = [scheduler_1_2, scheduler_1_2_sl, scheduler_beta_sl, scheduler_cogvideox]
# path = '/storage/dataset/mixkit-a-man-aggressively-yelling-at-a-woman-42260_resize1080p.mp4'
# transform = get_transform(args)
# video, fps = get_video(path, args.num_frames, transform, device, weight_dtype)

# bsz = video.shape[0]
# with torch.no_grad():
#     x = vae.encode(video) 
# # timesteps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
# timesteps = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

# init_seed = 1
# torch.manual_seed(init_seed)
# torch.cuda.manual_seed(init_seed)
# noise = torch.randn(1, 8, (args.num_frames-1)//4+1, args.max_height//8, args.max_width//8).to(device)
# videos_sche = []
# for scheduler in schedulers:
#     videos = []
#     for timestep in timesteps:
#         timestep = torch.LongTensor([timestep]*bsz).to(device)
#         latents = add_noise(x, noise, scheduler, timestep)
#         with torch.no_grad():
#             video = vae.decode(latents.to(vae.vae.dtype))
#         video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous() # b t h w c
#         videos.append(video)
#     videos = torch.cat(videos)
#     videos = save_video_grid(videos, nrow=1)
#     print(videos.shape)
#     videos_sche.append(videos)
# videos_sche = torch.stack(videos_sche)
# print(videos_sche.shape)
# videos_sche = save_video_grid(videos_sche, nrow=len(schedulers))
# print(videos_sche.shape)
# imageio.mimwrite(f'{args.num_frames}x{args.max_height}x{args.max_width}_addnoise_max{max(timesteps)}.mp4', videos_sche, fps=int(fps), quality=6) 





init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)

scheduler_cogvideox_0 = get_scheduler('cogvideox', 0.00085, 0.012, "scaled_linear", True, 0.25)
scheduler_cogvideox_1 = get_scheduler('cogvideox', 0.00085, 0.012, "scaled_linear", True, 1.0)
scheduler_cogvideox_2 = get_scheduler('cogvideox', 0.00085, 0.012, "scaled_linear", True, 4.0)
scheduler_cogvideox_3 = get_scheduler('cogvideox', 0.00085, 0.012, "scaled_linear", True, 6.0)
scheduler_cogvideox_4 = get_scheduler('cogvideox', 0.00085, 0.012, "scaled_linear", True, 9.0)
scheduler_cogvideox_5 = get_scheduler('cogvideox', 0.00085, 0.012, "scaled_linear", True, 24.0)
schedulers = [scheduler_cogvideox_0, scheduler_cogvideox_1, scheduler_cogvideox_2, 
              scheduler_cogvideox_3, scheduler_cogvideox_4, scheduler_cogvideox_5]
path = '/storage/dataset/mixkit-a-man-aggressively-yelling-at-a-woman-42260_resize1080p.mp4'

sizes = [[33, 320, 320], [33, 640, 640], [33, 960, 960]]  # snr 1 2 3

videos_size = []
for t, h, w in sizes:
    args.num_frames = t
    args.max_height = h
    args.max_width = w

    transform = get_transform(args)
    video, fps = get_video(path, args.num_frames, transform, device, weight_dtype)

    bsz = video.shape[0]
    with torch.no_grad():
        x = vae.encode(video) 
    # timesteps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    timesteps = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

    noise = torch.randn(1, 8, (args.num_frames-1)//4+1, args.max_height//8, args.max_width//8).to(device)
    videos_sche = []
    for scheduler in tqdm(schedulers):
        videos = []
        for timestep in timesteps:
            timestep = torch.LongTensor([timestep]*bsz).to(device)
            latents = add_noise(x, noise, scheduler, timestep)
            with torch.no_grad():
                video = vae.decode(latents.to(vae.vae.dtype))
            video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous() # b t h w c
            args.num_frames = 33
            args.max_height = 320
            args.max_width = 320
            transform_bak = get_transform(args)
            # b t h w c
            video_data = video[0].permute(0, 3, 1, 2)
            video = transform_bak(video_data)  # T C H W -> T C H W
            video = video.permute(0, 2, 3, 1)  # T C H W -> T H W C
            video = video.unsqueeze(0)
            video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8)
            videos.append(video)
        videos = torch.cat(videos)
        videos = save_video_grid(videos, nrow=1)
        videos_sche.append(videos)
    videos_sche = torch.stack(videos_sche)
    videos_sche = save_video_grid(videos_sche, nrow=len(schedulers))
    videos_size.append(videos_sche)
videos_size = torch.stack(videos_size)
print(videos_size.shape)
videos_size = save_video_grid(videos_size, nrow=len(sizes))
print(videos_size.shape)
imageio.mimwrite(f'snr0.25146924_addnoise_max{max(timesteps)}.mp4', videos_size, fps=int(fps), quality=6) 