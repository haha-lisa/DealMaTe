import torch
from PIL import Image
import argparse
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora
from diffusers import FluxPriorReduxPipeline
import os

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

def main(args):
    device = "cuda"
    base_path = "black-forest-labs/FLUX.1-dev"
    pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)

    material_image = Image.open(args.material_path).convert("RGB")

    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Redux-dev",
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe_prior_output = pipe_prior_redux(material_image)

    transformer = FluxTransformer2DModel.from_pretrained(
        base_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device=device
    )
    pipe.transformer = transformer
    pipe.to(device)

    lora_path = args.lora_path
    control_models = {
        "depth": f"{lora_path}/depth.safetensors",
        "normal": f"{lora_path}/normal.safetensors",
        "lighting": f"{lora_path}/lighting.safetensors",
    }

    paths = [control_models["depth"], control_models["normal"], control_models["lighting"]]
    set_multi_lora(pipe.transformer, paths, lora_weights=[[0], [1.2], [0.1]], cond_size=512)

    spatial_images = [
        Image.open(args.depth_path).convert("RGB"),
        Image.open(args.normal_path).convert("RGB"),
        Image.open(args.lighting_path).convert("RGB")
    ]

    image = pipe(
        **pipe_prior_output,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        spatial_images=spatial_images,
        cond_size=512,
    ).images[0]

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = os.path.splitext(os.path.basename(args.output_path))[0]
    generated_image_path = os.path.join(output_dir, f"{file_prefix}_generated.png")
    image.save(generated_image_path)

    content_image = Image.open(args.content_path).convert("RGB")
    mask_image = Image.open(args.mask_path).convert("L")
    mask_image = mask_image.resize(content_image.size, Image.LANCZOS)
    generated_image = Image.open(generated_image_path).convert("RGB")
    generated_image = generated_image.resize(content_image.size, Image.LANCZOS)

    result_image = Image.new("RGB", content_image.size)
    for x in range(content_image.width):
        for y in range(content_image.height):
            mask_pixel = mask_image.getpixel((x, y))
            if mask_pixel == 255:
                result_image.putpixel((x, y), generated_image.getpixel((x, y)))
            else:
                result_image.putpixel((x, y), content_image.getpixel((x, y)))

    result_image.save(args.output_path)
    print(f"Result saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DealMaTe: Multi-Dimensional Material Transfer")
    parser.add_argument("--material_path", required=True, help="Path to the material reference image")
    parser.add_argument("--content_path", required=True, help="Path to the content (target object) image")
    parser.add_argument("--mask_path", required=True, help="Path to the object mask image (white=object region)")
    parser.add_argument("--depth_path", required=True, help="Path to the depth map image")
    parser.add_argument("--lighting_path", required=True, help="Path to the lighting/shading image")
    parser.add_argument("--normal_path", required=True, help="Path to the surface normal image")
    parser.add_argument("--output_path", required=True, help="Path to save the final result image")
    parser.add_argument("--lora_path", default="./lora", help="Directory containing the LoRA weights (depth.safetensors, normal.safetensors, lighting.safetensors)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    main(args)
