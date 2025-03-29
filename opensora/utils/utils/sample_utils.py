from diffusers.schedulers import (
    DDIMScheduler, DDPMScheduler, PNDMScheduler,
    EulerDiscreteScheduler, DPMSolverMultistepScheduler,
    HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler, 
    DPMSolverSinglestepScheduler, CogVideoXDDIMScheduler
    )
from opensora.utils.scheduler import OpenSoraFlowMatchEulerScheduler
from einops import rearrange
import time
import torch
import os
import torch.distributed as dist
from torchvision.utils import save_image
import imageio
import math
import argparse
from transformers import AutoModelForCausalLM

try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import initialize_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
    pass

from opensora.utils.utils import set_seed
from opensora.models.causalvideovae import ae_stride_config, ae_wrapper
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.sample.pipeline_inpaint import OpenSoraInpaintPipeline
from opensora.models.diffusion.opensora_v1_3.modeling_opensora import OpenSoraT2V_v1_3
from opensora.models.diffusion.opensora_v1_3.modeling_inpaint import OpenSoraInpaint_v1_3
from opensora.models.text_encoder import get_text_tokenizer, get_text_cls

def get_scheduler(args):
    kwargs = dict(
        prediction_type=args.prediction_type, 
        rescale_betas_zero_snr=args.rescale_betas_zero_snr, 
        timestep_spacing="trailing" if args.rescale_betas_zero_snr else 'leading', 
    )
    if args.sample_method == 'DDIM':  
        scheduler_cls = DDIMScheduler
        kwargs['clip_sample'] = False
    elif args.sample_method == 'EulerDiscrete':
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_method == 'DDPM':  
        scheduler_cls = DDPMScheduler
        kwargs['clip_sample'] = False
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler_cls = DPMSolverMultistepScheduler
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_method == 'PNDM':
        scheduler_cls = PNDMScheduler
        kwargs.pop('rescale_betas_zero_snr', None)
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_method == 'DEISMultistep':
        scheduler_cls = DEISMultistepScheduler
        kwargs.pop('rescale_betas_zero_snr', None)
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    elif args.sample_method == 'CogVideoX':
        scheduler_cls = CogVideoXDDIMScheduler
    elif args.sample_method == 'OpenSoraFlowMatchEuler':
        scheduler_cls = OpenSoraFlowMatchEulerScheduler
        kwargs = {}
    else:
        raise NameError(f'Unsupport sample_method {args.sample_method}')
    scheduler = scheduler_cls(**kwargs)
    return scheduler

def prepare_pipeline(args, device):
    
    dtype_mapping = {
        'fp16': torch.float16, 
        'fp32': torch.float32, 
        'bf16': torch.bfloat16
    }
    
    ae_dtype = dtype_mapping[args.ae_dtype]
    weight_dtype = dtype_mapping[args.weight_dtype]

    vae = ae_wrapper[args.ae](args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=ae_dtype).eval()
    vae.vae_scale_factor = ae_stride_config[args.ae]
    if args.enable_tiling:
        vae.vae.enable_tiling()

    text_encoder_1 = get_text_cls(args.text_encoder_name_1).from_pretrained(
        args.text_encoder_name_1, cache_dir=args.cache_dir, 
        torch_dtype=weight_dtype
        ).eval()
    tokenizer_1 = get_text_tokenizer(args.text_encoder_name_1).from_pretrained(
        args.text_encoder_name_1, cache_dir=args.cache_dir
        )

    if args.text_encoder_name_2 is not None:
        text_encoder_2 = get_text_cls(args.text_encoder_name_2).from_pretrained(
            args.text_encoder_name_2, cache_dir=args.cache_dir, 
            torch_dtype=weight_dtype
            ).eval()
        tokenizer_2 = get_text_tokenizer(args.text_encoder_name_2).from_pretrained(
            args.text_encoder_name_2, cache_dir=args.cache_dir
            )
    else:
        text_encoder_2, tokenizer_2 = None, None
        
    if args.text_encoder_name_3 is not None:
        text_encoder_3 = get_text_cls(args.text_encoder_name_3).from_pretrained(
            args.text_encoder_name_3, cache_dir=args.cache_dir, 
            torch_dtype=weight_dtype
            ).eval()
        tokenizer_3 = get_text_tokenizer(args.text_encoder_name_3).from_pretrained(
            args.text_encoder_name_3, cache_dir=args.cache_dir
            )
    else:
        text_encoder_3, tokenizer_3 = None, None

    if args.version == 'v1_3':
        if args.model_type == 'inpaint' or args.model_type == 'i2v':
            transformer_model = OpenSoraInpaint_v1_3.from_pretrained(
                args.model_path, cache_dir=args.cache_dir,
                device_map=None, torch_dtype=weight_dtype
                ).eval()
        else:
            transformer_model = OpenSoraT2V_v1_3.from_pretrained(
                args.model_path, cache_dir=args.cache_dir,
                device_map=None, torch_dtype=weight_dtype
                ).eval()
    elif args.version == 'v1_5':
        if args.model_type == 'inpaint' or args.model_type == 'i2v':
            raise NotImplementedError('Inpainting model is not available in v1_5')
        else:
            from opensora.models.diffusion.opensora_v1_5.modeling_opensora import OpenSoraT2V_v1_5
            transformer_model = OpenSoraT2V_v1_5.from_pretrained(
                args.model_path, cache_dir=args.cache_dir, 
                # device_map=None, 
                torch_dtype=weight_dtype
                ).eval()
    elif args.version == 't2i':
        if args.model_type == 'inpaint' or args.model_type == 'i2v':
            raise NotImplementedError('Inpainting model is not available in t2i')
        else:
            from opensora.models.diffusion.opensora_t2i.modeling_opensora import OpenSoraT2I
            transformer_model = OpenSoraT2I.from_pretrained(
                args.model_path, cache_dir=args.cache_dir, 
                # device_map=None, 
                torch_dtype=weight_dtype
                ).eval()
    scheduler = get_scheduler(args)
    pipeline_class = OpenSoraInpaintPipeline if args.model_type == 'inpaint' or args.model_type == 'i2v' else OpenSoraPipeline

    pipeline = pipeline_class(
        vae=vae,
        text_encoder=text_encoder_1,
        tokenizer=tokenizer_1,
        scheduler=scheduler,
        transformer=transformer_model, 
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        text_encoder_3=text_encoder_3,
        tokenizer_3=tokenizer_3,
    ).to(device)

    if args.save_memory:
        print('enable_model_cpu_offload AND enable_sequential_cpu_offload AND enable_tiling')
        pipeline.enable_model_cpu_offload()
        pipeline.enable_sequential_cpu_offload()
        # torch.cuda.empty_cache()
        vae.vae.enable_tiling()
        vae.vae.t_chunk_enc = 8
        vae.vae.t_chunk_dec = vae.vae.t_chunk_enc // 2
        
    if args.compile:
        pipeline.transformer = torch.compile(pipeline.transformer)

    return pipeline

def init_gpu_env(args):
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', 
        world_size=world_size, rank=local_rank
        )
    if args.sp:
        initialize_sequence_parallel_state(world_size)
    return args

def init_npu_env(args):
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(
        backend='hccl', init_method='env://', 
        world_size=world_size, rank=local_rank
        )
    if args.sp:
        initialize_sequence_parallel_state(world_size)
    return args


def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros(
        (
            t, 
            (padding + h) * nrow + padding, 
            (padding + w) * ncol + padding, 
            c
        ), 
        dtype=torch.uint8
        )

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    return video_grid


def run_model_and_save_samples(args, pipeline, caption_refiner_model=None, enhance_video_model=None):

    if args.seed is not None:
        set_seed(args.seed, rank=args.local_rank, device_specific=True)
    if args.local_rank >= 0:
        torch.manual_seed(args.seed + args.local_rank)
    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    video_grids = []
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]
    
    if args.model_type == 'inpaint' or args.model_type == 'i2v':
        if not isinstance(args.conditional_pixel_values_path, list):
            args.conditional_pixel_values_path = [args.conditional_pixel_values_path]
        if len(args.conditional_pixel_values_path) == 1 and args.conditional_pixel_values_path[0].endswith('txt'):
            temp = open(args.conditional_pixel_values_path[0], 'r').readlines()
            conditional_pixel_values_path = [i.strip().split(',') for i in temp]
        
        mask_type = args.mask_type if args.mask_type is not None else None

    if args.use_pos_neg_prompt:
        positive_prompt = """
        high quality, high aesthetic, {}
        """

        negative_prompt = """
        nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
        low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
        """
    else:
        positive_prompt = "{}"
        negative_prompt = ""
    
    def generate(prompt, conditional_pixel_values_path=None, mask_type=None):
        
        if args.caption_refiner is not None:
            if args.model_type != 'inpaint' and args.model_type != 'i2v':
                refine_prompt = caption_refiner_model.get_refiner_output(prompt)
                print(f'\nOrigin prompt: {prompt}\n->\nRefine prompt: {refine_prompt}')
                prompt = refine_prompt
            else:
                # Due to the current use of LLM as the caption refiner, additional content that is not present in the control image will be added. Therefore, caption refiner is not used in this mode.
                print('Caption refiner is not available for inpainting model, use the original prompt...')
                time.sleep(3)
        input_prompt = positive_prompt.format(prompt)
        
        if args.model_type == 'inpaint' or args.model_type == 'i2v':
            print(f'\nConditional pixel values path: {conditional_pixel_values_path}')
            videos = pipeline(
                conditional_pixel_values_path=conditional_pixel_values_path,
                mask_type=mask_type,
                crop_for_hw=args.crop_for_hw,
                max_hxw=args.max_hxw,
                prompt=input_prompt, 
                negative_prompt=negative_prompt, 
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_rescale=args.guidance_rescale, 
                guidance_scale=args.guidance_scale,
                num_samples_per_prompt=args.num_samples_per_prompt,
                max_sequence_length=args.max_sequence_length,
                use_linear_quadratic_schedule=args.use_linear_quadratic_schedule, 
                first_linear_monitor_step=args.first_linear_monitor_step,
                second_linear_monitor_step=args.second_linear_monitor_step,
                third_linear_monitor_step=args.third_linear_monitor_step,
                pivot=args.pivot,
                pivot_1=args.pivot_1,
                use_two_linear_schedule=args.use_two_linear_schedule, 
                use_three_linear_schedule=args.use_three_linear_schedule, 
            ).videos
        else:
            videos = pipeline(
                input_prompt, 
                negative_prompt=negative_prompt, 
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_rescale=args.guidance_rescale, 
                guidance_scale=args.guidance_scale,
                num_samples_per_prompt=args.num_samples_per_prompt,
                max_sequence_length=args.max_sequence_length,
                use_linear_quadratic_schedule=args.use_linear_quadratic_schedule, 
                first_linear_monitor_step=args.first_linear_monitor_step,
                second_linear_monitor_step=args.second_linear_monitor_step,
                third_linear_monitor_step=args.third_linear_monitor_step,
                pivot=args.pivot,
                pivot_1=args.pivot_1,
                use_two_linear_schedule=args.use_two_linear_schedule, 
                use_three_linear_schedule=args.use_three_linear_schedule, 
            ).videos
        if enhance_video_model is not None:
            # b t h w c
            videos = enhance_video_model.enhance_a_video(videos, input_prompt, 2.0, args.fps, 250)
        if (not args.sp) or (args.sp and args.local_rank <= 0):
            if args.num_frames == 1:
                videos = rearrange(videos, 'b t h w c -> (b t) c h w')
                if args.num_samples_per_prompt != 1:
                    for i, image in enumerate(videos):
                        save_image(
                            image / 255.0, 
                            os.path.join(
                                args.save_img_path, 
                                f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}_i{i}.jpg'
                                ),
                            nrow=math.ceil(math.sqrt(videos.shape[0])), 
                            normalize=True, 
                            value_range=(0, 1)
                            )  # b c h w
                save_image(
                    videos / 255.0, 
                    os.path.join(
                        args.save_img_path, 
                        f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.jpg'
                        ),
                    nrow=math.ceil(math.sqrt(videos.shape[0])), 
                    normalize=True, 
                    value_range=(0, 1)
                    )  # b c h w
            else:
                if args.num_samples_per_prompt == 1:
                    imageio.mimwrite(
                        os.path.join(
                            args.save_img_path,
                            f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4'
                        ), 
                        videos[0],  # 1 t h w c
                        fps=args.fps, 
                        quality=6
                        )  # highest quality is 10, lowest is 0
                else:
                    for i in range(args.num_samples_per_prompt):
                        imageio.mimwrite(
                            os.path.join(
                                args.save_img_path,
                                f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}_i{i}.mp4'
                            ), videos[i],  # t h w c
                            fps=args.fps, 
                            quality=6
                            )  # highest quality is 10, lowest is 0
                        
                    videos = save_video_grid(videos)
                    imageio.mimwrite(
                        os.path.join(
                            args.save_img_path,
                            f'{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4'
                        ), 
                        videos,
                        fps=args.fps, 
                        quality=6
                        )  # highest quality is 10, lowest is 0)
                    videos = videos.unsqueeze(0) # 1 t h w c
            video_grids.append(videos)

    if args.model_type == 'inpaint' or args.model_type == 'i2v':
        for index, (prompt, cond_path) in enumerate(zip(args.text_prompt, conditional_pixel_values_path)):
            if not args.sp and args.local_rank != -1 and index % args.world_size != args.local_rank:
                continue
            generate(prompt, cond_path, mask_type)
    else:
        for index, prompt in enumerate(args.text_prompt):
            if not args.sp and args.local_rank != -1 and index % args.world_size != args.local_rank:
                continue  # skip when ddp
            generate(prompt)

    if (args.model_type == "inpaint" or args.model_type == "i2v") and not args.crop_for_hw:
        print('completed, please check the saved images and videos')
    else:
        if not args.sp:
            if args.local_rank != -1:
                dist.barrier()
                assert len(args.text_prompt) >= args.world_size
                video_grids = torch.cat(video_grids, dim=0).cuda()  # num c h w or 1 t h w c
                
                shape = list(video_grids.shape)  # num c h w or 1 t h w c
                if args.num_frames == 1:
                    max_sample = math.ceil(len(args.text_prompt) / args.world_size) * args.num_samples_per_prompt  # max = 8
                else:
                    max_sample = math.ceil(len(args.text_prompt) / args.world_size)
                video_grids_to_gather = [torch.zeros(*shape[1:], dtype=torch.uint8).cuda() for _ in range(max_sample)]
                # true video are filled to video_grids_to_gather, maybe the last element is all zero.
                for i, v in enumerate(video_grids):
                    video_grids_to_gather[i] = v
                video_grids_to_gather = torch.stack(video_grids_to_gather, dim=0)

                shape[0] = max_sample * args.world_size
                gathered_tensor = torch.zeros(shape, dtype=torch.uint8).cuda()
                dist.all_gather_into_tensor(gathered_tensor, video_grids_to_gather.contiguous())
                video_grids = gathered_tensor.cpu()
                which_to_save = torch.sum(video_grids, dim=list(range(video_grids.ndim))[1:]).bool()
                video_grids = video_grids[which_to_save]
                dist.barrier()
            else:
                video_grids = torch.cat(video_grids, dim=0)
        elif args.sp and args.local_rank <= 0:
            video_grids = torch.cat(video_grids)
        
        if args.local_rank <= 0:
            if args.num_frames == 1:
                save_image(
                    video_grids / 255.0, 
                    os.path.join(
                        args.save_img_path,
                        f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.jpg'
                        ), 
                    nrow=math.ceil(math.sqrt(len(video_grids))), 
                    normalize=True, 
                    value_range=(0, 1)
                    )
            else:
                video_grids = save_video_grid(video_grids)
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path,
                        f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4'
                    ), 
                    video_grids, 
                    fps=args.fps, 
                    quality=6
                    )
            print('save path {}'.format(args.save_img_path))



def run_model_and_save_samples_npu(args, pipeline, caption_refiner_model=None, enhance_video_model=None):
    
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    #     aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization
    # )
    # profile_output_path = "/home/image_data/npu_profiling_t2v"
    # os.makedirs(profile_output_path, exist_ok=True)
    # with torch_npu.profiler.profile(
    #         activities=[
    #             torch_npu.profiler.ProfilerActivity.NPU, 
    #             torch_npu.profiler.ProfilerActivity.CPU
    #             ],
    #         with_stack=True,
    #         record_shapes=True,
    #         profile_memory=True,
    #         experimental_config=experimental_config,
    #         schedule=torch_npu.profiler.schedule(
    #             wait=10000, warmup=0, active=1, repeat=1, skip_first=0
    #             ),
    #         on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{profile_output_path}/")
    # ) as prof:
    run_model_and_save_samples(args, pipeline, caption_refiner_model, enhance_video_model)
        # prof.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default='v1_3', choices=['v1_3', 'v1_5', 't2i'])
    parser.add_argument("--model_type", type=str, default='t2v', choices=['t2v', 'inpaint', 'i2v'])
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ae_dtype", type=str, default='fp16')
    parser.add_argument("--weight_dtype", type=str, default='fp16')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--caption_refiner", type=str, default=None)
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--enhance_video", type=str, default=None)
    parser.add_argument("--text_encoder_name_1", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--text_encoder_name_3", type=str, default=None)
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples_per_prompt", type=int, default=1)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--refine_caption', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true') 
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
    parser.add_argument('--rescale_betas_zero_snr', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)    
    parser.add_argument('--world_size', type=int, default=1)    
    parser.add_argument('--sp', action='store_true')
    parser.add_argument('--use_pos_neg_prompt', action='store_true')

    parser.add_argument('--first_linear_monitor_step', type=int, default=100)    
    parser.add_argument('--second_linear_monitor_step', type=int, default=0)   
    parser.add_argument('--third_linear_monitor_step', type=int, default=0)    
    parser.add_argument('--pivot', type=float, default=0.1)    
    parser.add_argument('--pivot_1', type=float, default=0.1)    
    parser.add_argument('--use_two_linear_schedule', action='store_true')
    parser.add_argument('--use_three_linear_schedule', action='store_true')

    parser.add_argument('--use_linear_quadratic_schedule', action='store_true')
    parser.add_argument('--conditional_pixel_values_path', type=str, default=None)
    parser.add_argument('--mask_type', type=str, default=None)
    parser.add_argument('--crop_for_hw', action='store_true')
    parser.add_argument('--max_hxw', type=int, default=236544) # 480*480
    args = parser.parse_args()
    assert not (args.sp and args.num_frames == 1)
    return args