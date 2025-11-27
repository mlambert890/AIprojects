class Tool:
    functions = [
        {
            "name": "comfy_generate",
            "description": "Generate images using ComfyUI with advanced parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "steps": {"type": "integer"},
                    "cfg_scale": {"type": "number"},
                    "sampler": {"type": "string"},
                    "checkpoint": {"type": "string"},
                    "seed": {"type": "integer"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "batch_size": {"type": "integer"},
                    "comfy_base_url": {"type": "string"}
                },
                "required": ["prompt"]
            }
        }
    ]

    async def comfy_generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 20,
        cfg_scale: float = 7.5,
        sampler: str = "euler",
        checkpoint: str = "model.safetensors",
        seed: int = None,
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        comfy_base_url: str = "http://127.0.0.1:8188"
    ):
        import aiohttp
        import base64
        import random

        if seed is None:
            seed = random.randint(1, 2**31 - 1)

        # ComfyUI workflow graph assembly
        workflow = {
            "1": {  # Load checkpoint
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint}
            },
            "2": {  # Encode positive prompt
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1]
                }
            },
            "3": {  # Encode negative prompt
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1]
                }
            },
            "4": {  # Latent noise
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": batch_size
                }
            },
            "5": {  # Sampler
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": sampler,
                    "scheduler": "normal",
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "6": {  # Decode VAE
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "7": {  # Save image
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0]
                }
            }
        }

        # Submit workflow to ComfyUI
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{comfy_base_url}/prompt",
                json={"prompt": workflow}
            ) as resp:
                if resp.status != 200:
                    return {"error": f"HTTP {resp.status} submitting workflow"}
                data = await resp.json()

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            return {"error": "No prompt_id returned by ComfyUI."}

        # Poll the history endpoint until results appear
        images = []
        history_url = f"{comfy_base_url}/history/{prompt_id}"

        for _ in range(120):  # 120 × 1s = 2 minutes max
            async with aiohttp.ClientSession() as session:
                async with session.get(history_url) as r:
                    if r.status == 200:
                        history = await r.json()
                        outputs = list(history.get(prompt_id, {})
                                              .get("outputs", {})
                                              .values())

                        if outputs:
                            images = outputs[0]["images"]
                            break

            import asyncio
            await asyncio.sleep(1)

        if not images:
            return {"error": "Timed out waiting for ComfyUI output."}

        # Fetch actual image binaries
        base64_outputs = []
        async with aiohttp.ClientSession() as session:
            for img in images:
                filename = img["filename"]
                subfolder = img.get("subfolder", "")
                type_ = img.get("type", "output")
                url = f"{comfy_base_url}/view?filename={filename}&subfolder={subfolder}&type={type_}"

                async with session.get(url) as img_resp:
                    if img_resp.status == 200:
                        binary = await img_resp.read()
                        b64 = base64.b64encode(binary).decode("utf-8")
                        base64_outputs.append(b64)

        return {
            "images": base64_outputs,
            "metadata": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler": sampler,
                "checkpoint": checkpoint,
                "seed": seed,
                "width": width,
                "height": height,
                "batch_size": batch_size
            }
        }
