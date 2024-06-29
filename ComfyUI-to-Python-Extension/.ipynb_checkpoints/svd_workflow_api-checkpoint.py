import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import VAEDecode, NODE_CLASS_MAPPINGS, VAELoader, KSampler, LoadImage


def main():
    import_custom_nodes()
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_23 = loadimage.load_image(
            image="RF_Stocksy_txpa13c4dc9Xtv300_Medium_979569.jpg"
        )

        cm_nearestsdxlresolution = NODE_CLASS_MAPPINGS["CM_NearestSDXLResolution"]()
        cm_nearestsdxlresolution_84 = cm_nearestsdxlresolution.op(
            image=get_value_at_index(loadimage_23, 0)
        )

        imageonlycheckpointloader = NODE_CLASS_MAPPINGS["ImageOnlyCheckpointLoader"]()
        imageonlycheckpointloader_15 = imageonlycheckpointloader.load_checkpoint(
            ckpt_name="svd_xt.safetensors"
        )

        vaeloader = VAELoader()
        vaeloader_89 = vaeloader.load_vae(vae_name="vae-ft-mse-840000-ema-pruned.ckpt")

        svd_img2vid_conditioning = NODE_CLASS_MAPPINGS["SVD_img2vid_Conditioning"]()
        svd_img2vid_conditioning_12 = svd_img2vid_conditioning.encode(
            width=get_value_at_index(cm_nearestsdxlresolution_84, 0),
            height=get_value_at_index(cm_nearestsdxlresolution_84, 1),
            video_frames=6,
            motion_bucket_id=40,
            fps=12,
            augmentation_level=0.05,
            clip_vision=get_value_at_index(imageonlycheckpointloader_15, 1),
            init_image=get_value_at_index(loadimage_23, 0),
            vae=get_value_at_index(vaeloader_89, 0),
        )

        cr_seed = NODE_CLASS_MAPPINGS["CR Seed"]()
        cr_seed_90 = cr_seed.seedint(seed=random.randint(1, 2**64))

        videolinearcfgguidance = NODE_CLASS_MAPPINGS["VideoLinearCFGGuidance"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(1):
            videolinearcfgguidance_14 = videolinearcfgguidance.patch(
                min_cfg=1, model=get_value_at_index(imageonlycheckpointloader_15, 0)
            )

            ksampler_38 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=4,
                cfg=5.5,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(videolinearcfgguidance_14, 0),
                positive=get_value_at_index(svd_img2vid_conditioning_12, 0),
                negative=get_value_at_index(svd_img2vid_conditioning_12, 1),
                latent_image=get_value_at_index(svd_img2vid_conditioning_12, 2),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_38, 0),
                vae=get_value_at_index(vaeloader_89, 0),
            )

            vhs_videocombine_26 = vhs_videocombine.combine_video(
                frame_rate=12,
                loop_count=0,
                filename_prefix="SVD-xt",
                format="video/h264-mp4",
                pingpong=False,
                save_output=True,
                images=get_value_at_index(vaedecode_8, 0),
                unique_id=17174442833647817766,
            )


if __name__ == "__main__":
    main()
