import argparse

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from torchvision.utils import save_image
import os
import math
from accelerate import Accelerator
from accelerate.utils import set_seed
from opensora.eval.inversion.pipeline_inversion import OpenSoraInversionPipeline
from opensora.eval.inversion.inversion_dataset import InversionValidImageDataset
from opensora.utils.sample_utils import OpenSoraFlowMatchEulerScheduler
from opensora.models.causalvideovae import (
    WFVAEModelWrapper,
    ae_stride_config,
    ae_wrapper,
)
from opensora.models.text_encoder import get_text_tokenizer, get_text_cls
from opensora.models.diffusion.opensora_v1_5.modeling_opensora import OpenSoraT2V_v1_5


def main(args):
    # Prepare environment
    accelerator = Accelerator()
    set_seed(1234, device_specific=False)  # every process has the same seed
    device = accelerator.device
    weight_dtype = accelerator.mixed_precision
    if weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    os.makedirs(args.save_img_path, exist_ok=True)

    # Load VAE
    vae = ae_wrapper[args.ae](args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype).eval()
    vae.vae_scale_factor = ae_stride_config[args.ae]
    if args.enable_tiling:
        vae.vae.enable_tiling()

    # Load text encoders
    text_encoder_1 = (
        get_text_cls(args.text_encoder_name_1)
        .from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir, torch_dtype=weight_dtype
        )
        .eval()
    )
    tokenizer_1 = get_text_tokenizer(args.text_encoder_name_1).from_pretrained(
        args.text_encoder_name_1, cache_dir=args.cache_dir
    )

    if args.text_encoder_name_2:
        text_encoder_2 = (
            get_text_cls(args.text_encoder_name_2)
            .from_pretrained(
                args.text_encoder_name_2,
                cache_dir=args.cache_dir,
                torch_dtype=weight_dtype,
            )
            .eval()
        )
        tokenizer_2 = get_text_tokenizer(args.text_encoder_name_2).from_pretrained(
            args.text_encoder_name_2, cache_dir=args.cache_dir
        )
    else:
        text_encoder_2, tokenizer_2 = None, None

    if args.text_encoder_name_3:
        text_encoder_3 = (
            get_text_cls(args.text_encoder_name_3)
            .from_pretrained(
                args.text_encoder_name_3,
                cache_dir=args.cache_dir,
                torch_dtype=weight_dtype,
            )
            .eval()
        )
        tokenizer_3 = get_text_tokenizer(args.text_encoder_name_3).from_pretrained(
            args.text_encoder_name_3, cache_dir=args.cache_dir
        )
    else:
        text_encoder_3, tokenizer_3 = None, None

    # Load diffusion model
    if args.version == 'v1_5':
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
    else:
        raise Exception(
            "Failed to load diffusion model. Please check your model version, inversion validation is now only supported for OpenSoraT2V_v1_5."
        )

    # Load scheduler
    scheduler = OpenSoraFlowMatchEulerScheduler()

    # Prepare pipeline
    pipeline = OpenSoraInversionPipeline(
        vae=vae,
        transformer=transformer_model,
        scheduler=scheduler,
        text_encoder=text_encoder_1,
        tokenizer=tokenizer_1,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        text_encoder_3=text_encoder_3,
        tokenizer_3=tokenizer_3,
    ).to(device=device)

    # Prepare dataset
    dataset = InversionValidImageDataset(args)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False
    )

    # Prepare accelerator
    dataloader = accelerator.prepare(dataloader)
    # Run pipeline
    # final_inversion_loss = {
    #     num_inverse_steps: [] for num_inverse_steps in args.num_inverse_steps
    # }
    for batch in tqdm(dataloader):
        cache_dict = {}
        for num_inverse_steps in args.num_inverse_steps:
            videos = pipeline(
                image=batch["image"], # b c 1 h w
                prompt=batch["caption"],
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                num_inverse_steps=num_inverse_steps,
                guidance_scale=args.guidance_scale,
                guidance_rescale=args.guidance_rescale,
                num_samples_per_prompt=args.num_samples_per_prompt,
                max_sequence_length=args.max_sequence_length,
                use_linear_quadratic_schedule=False,
                inverse_cache_dict=cache_dict
            ).videos
            videos = rearrange(videos, "b t h w c -> (b t) c h w")

            for v, rela_path in zip(videos, batch["rela_path"]):
                rela_path = rela_path.replace('.jpg', f'_{num_inverse_steps}.png')
                save_path = os.path.join(args.save_img_path, rela_path)
                directory_path = os.path.dirname(save_path)
                os.makedirs(directory_path, exist_ok=True)
                save_image(
                    v / 255.0,
                    save_path,
                    nrow=math.ceil(math.sqrt(v.shape[0])),
                    normalize=True,
                    value_range=(0, 1),
                )  # b c h w
            
            # mse = F.mse_loss(videos.cpu().float() / 255.0, (batch["image"].cpu().float() / 2 + 0.5))
            # print(num_inverse_steps, mse)
            # final_inversion_loss[num_inverse_steps].append(mse)
        
        batch["image"] = rearrange(batch["image"], "b c t h w -> (b t) c h w")
        for v, rela_path in zip(batch["image"], batch["rela_path"]):
            rela_path = rela_path.replace('.jpg', f'_gt.png')
            save_path = os.path.join(args.save_img_path, rela_path)
            # directory_path = os.path.dirname(save_path)
            # os.makedirs(directory_path, exist_ok=True)
            save_image(
                v.cpu() / 2 + 0.5,
                save_path,
                nrow=math.ceil(math.sqrt(v.shape[0])),
                normalize=True,
                value_range=(0, 1),
            )  # b c h w

    # accelerator.wait_for_everyone()
    # final_inversion_loss = (
    #     accelerator.gather_for_metrics(final_inversion_loss)
    # )
    # print(final_inversion_loss)
    # if accelerator.is_main_process:
    #     print("=" * 50)
    #     print("Inversion Loss Results")
    #     print("=" * 50)
    #     print("Steps | Mean Loss")
    #     print("-" * 50)
    #     for num_inverse_steps in args.num_inverse_steps:
    #         mean_loss = np.mean(final_inversion_loss[num_inverse_steps])
    #         print(f"{num_inverse_steps:5d} | {mean_loss:.6f}")
    #     print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="model path")
    parser.add_argument("--model_type", type=str, default='t2v', choices=['t2v', 'inpaint', 'i2v'])
    parser.add_argument("--version", type=str, default='t2i', choices=['v1_5', 't2i'])
    parser.add_argument(
        "--ae", type=str, default="WFVAEModel_D32_8x8x8", help="ae name"
    )
    parser.add_argument("--ae_path", type=str, default="", help="ae path")
    parser.add_argument(
        "--text_encoder_name_1",
        type=str,
        default="",
        help="text encoder name 1",
    )
    parser.add_argument(
        "--text_encoder_name_2", type=str, default="", help="text encoder name 2"
    )
    parser.add_argument(
        "--text_encoder_name_3",
        type=str,
        default="",
        help="text encoder name 3",
    )
    parser.add_argument("--no_condition", action="store_true", help="enable tiling")
    parser.add_argument("--enable_tiling", action="store_true", help="enable tiling")
    parser.add_argument("--data_root", type=str, default="", help="data root")
    parser.add_argument("--data_path", type=str, default="", help="data path")
    parser.add_argument("--num_samples", type=int, default=-1, help="num samples")
    parser.add_argument("--save_path", type=str, default="", help="save path")
    parser.add_argument(
        "--num_inference_steps", type=int, default=100, help="num inference steps"
    )
    parser.add_argument(
        "--num_inverse_steps",
        type=int,
        nargs="+",
        default=[10, 30, 50, 70],
        help="num inverse steps",
    )
    parser.add_argument("--height", type=int, default=256, help="resolution")
    parser.add_argument("--width", type=int, default=256, help="resolution")
    parser.add_argument("--num_frames", type=int, default=1, help="num frames")
    parser.add_argument(
        "--guidance_scale", type=float, default=7.0, help="guidance scale"
    )
    parser.add_argument(
        "--guidance_rescale", type=float, default=0.7, help="guidance scale"
    )
    parser.add_argument(
        "--num_samples_per_prompt", type=int, default=1, help="num samples per prompt"
    )
    parser.add_argument(
        "--max_sequence_length", type=int, default=512, help="max sequence length"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="num workers")
    parser.add_argument(
        "--cache_dir", type=str, default="./cache_dir", help="cache dir"
    )
    parser.add_argument(
        "--save_img_path", type=str, default="./test_inversion", help="save img path"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
