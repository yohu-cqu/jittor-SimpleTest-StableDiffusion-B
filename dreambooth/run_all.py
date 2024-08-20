import os, sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "/home/user1/jittor2024/JDiffusion/cached_path"

import json, tqdm, torch
import jittor as jt
jt.flags.use_rocm = 1

from JDiffusion.pipelines import StableDiffusionPipeline

root="/home/user1/jittor2024/jittor-B-commit"
save_root = f"{root}/dreambooth/results/" + "prompt_v1_cosine_test"

max_num = 28
inference_steps = 200
checkpoint_epoch = 300
seed = 76587
jt.set_global_seed(seed)
dataset_root = f"{root}/B/"
style_file = f"{root}/dreambooth/settings/style.json"
texture_file = f"{root}/dreambooth/settings/texture.json"
color_file = f"{root}/dreambooth/settings/color.json"

with open(style_file, "r") as f:
    style_dict = json.load(f)

with open(texture_file, "r") as f:
    texture_dict = json.load(f)

with open(color_file, "r") as f:
    color_dict = json.load(f)

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
with torch.no_grad():
    for tempid in tqdm.tqdm(range(0, max_num)):
        taskid = "{:0>2d}".format(tempid)
        if checkpoint_epoch is None:
            pipe.load_lora_weights(os.path.join(save_root, f"style/style_{taskid}"))
        else:
            pipe.load_lora_weights(os.path.join(save_root, f"style_{checkpoint_epoch}epoch/style_{taskid}"))

        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            new_prompt = f"A photo of {prompt} in {style_dict[taskid]} style, with a texture of {texture_dict[taskid]} and with a color style of {color_dict[taskid]}."
            print(new_prompt)
            image = pipe(prompt=new_prompt, num_inference_steps=inference_steps, width=512, height=512, seed=seed).images[0]
            print(os.path.join(save_root, f"outputs_{checkpoint_epoch}ckpt_{inference_steps}steps_{seed}seed/{taskid}"))
            os.makedirs(os.path.join(save_root, f"outputs_{checkpoint_epoch}ckpt_{inference_steps}steps_{seed}seed/{taskid}"), exist_ok=True)
            image.save(os.path.join(save_root, f"outputs_{checkpoint_epoch}ckpt_{inference_steps}steps_{seed}seed/{taskid}/{prompt}.png"))
