'''
https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl
https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl
https://github.com/damian0815/compel
'''
from compel import Compel, ReturnedEmbeddingsType
from datetime import datetime
from diffusers import StableDiffusionXLPipeline
import argparse
import json
import gc
import random
import torch


def do_diffusion(args):
    pipeline = StableDiffusionXLPipeline.from_single_file(
        args.checkpoint_file,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        requires_safety_checker=False).to("cuda")

    if args.scheduler == "EulerDiscreteScheduler":
        from diffusers import EulerDiscreteScheduler
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config)
    if args.scheduler == "EulerAncestralDiscreteScheduler":
        from diffusers import EulerAncestralDiscreteScheduler
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config)
    elif args.scheduler == "DPMSolverMultistepScheduler":
        from diffusers import DPMSolverMultistepScheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config, use_karras_sigmas=True)
    elif args.scheduler == "DPMSolverMultistepScheduler2MSDEKarras":
        from diffusers import DPMSolverMultistepScheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++")

    if args.lora1_file:
        pipeline.load_lora_weights(args.lora1_file, adapter_name="lora1")
    if args.lora2_file:
        pipeline.load_lora_weights(args.lora2_file, adapter_name="lora2")

    embeddings_type = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
    if args.clip_skip == 2:
        embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED

    compel = Compel(
        tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
        text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
        returned_embeddings_type=embeddings_type,
        requires_pooled=[False, True])

    seed = random.randint(0, 1000000)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Get prompt from user input in a loop
    while True:
        try:
            options = input("Enter prompt: ")
            if options == "quit":
                break
            else:
                options = json.loads(options)

            lora1_scale = float(options.get('lora1_scale', 0.0))
            lora2_scale = float(options.get('lora2_scale', 0.0))

            # LORAs must load before conditioning
            if lora1_scale > 0 or lora2_scale > 0:
                pipeline.set_adapters(
                    ["lora1", "lora2"],
                    adapter_weights=[lora1_scale, lora2_scale])
            else:
                pipeline.disable_lora()

            embeds, pooled = compel(options['prompt'])
            negative_embeds, negative_pooled = compel(
                options['negative_prompt'])

            image = pipeline(prompt_embeds=embeds,
                             pooled_prompt_embeds=pooled,
                             negative_prompt_embeds=negative_embeds,
                             negative_pooled_prompt_embeds=negative_pooled,
                             height=int(options['height']),
                             width=int(options['width']),
                             guidance_scale=float(options['scale']),
                             num_inference_steps=int(options['num_steps']),
                             generator=generator).images[0]
            date_time = datetime.now().strftime("%Y%m%d%H%M%S")
            image.save(args.output_directory + '/' + date_time + '.png')

            # Clean up
            del image
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(e)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--clip_skip", type=int, required=True)
    parser.add_argument("--lora1_file", type=str, required=False)
    parser.add_argument("--lora2_file", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = load_args()
    do_diffusion(args)
