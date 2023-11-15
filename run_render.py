import glob
import os

import fire
from loguru import logger
from multiprocessing import Pool
import subprocess


def render_objects(
    object_path: str = "sample_data",
    output_dir: str = "sample_output",
    three_views: bool = True,
    num_renders: int = 5,
    num_trials: int = 5,
    freestyle: bool = True,
    visible_edges: bool = False,
    engine: str = "CYCLES",
    only_northern_hemisphere: bool = False,
    render_size: int = 1024,
    res_percentage: int = 100,
    output_channels: str = "RGBA",
) -> bool:
    """Render the objects in the raw data path.
    Args:
    Returns: True if the object was rendered successfully, False otherwise.
    """
    args = ""
    # check if we should only render the northern hemisphere
    if only_northern_hemisphere:
        args += "--only_northern_hemisphere"

    # get all objects in the raw data path
    all_objects = glob.glob("**/*.stl", root_dir=object_path, recursive=True)
    logger.info(f"Found {len(all_objects)} objects to render in {object_path}.")

    # set render style
    if freestyle:
        args += " --freestyle"
    args += f" --engine {engine}"
    args += f" --render_size {render_size}"
    args += f" --res_percentage {res_percentage}"
    if three_views:
        args += " --three_views"
        if visible_edges:
            args += " --visible_edges"
        args += f" --camera_type ORTHO"
    args += f" --num_renders {num_renders}"
    args += f" --num_trials {num_trials}"
    args += f" --output_channels {output_channels}"
    command = f"python blender.py {args}"

    # get (input, output) paths
    io_paths = [
        (
            os.path.join(object_path, obj),
            os.path.join(output_dir, os.path.basename(obj).split(".")[0]),
        )
        for obj in all_objects
    ]
    commands = [
        command + f" --object_path {obj} --output_dir {output_dir}"
        for obj, output_dir in io_paths
    ]

    cpu_count = os.cpu_count()
    p_count = 4
    if len(all_objects) > cpu_count:
        # render all objects in parallel
        # get cpu count
        with Pool(processes=p_count) as p:
            p.map(subprocess_cmd, commands)
    else:
        # render all objects sequentially
        for command in commands:
            subprocess_cmd(command)

    logger.info(f"Done rendering {len(all_objects)} objects.")


def subprocess_cmd(command):
    # get object name
    object_name = command.split("--object_path")[1].split("--")[0].strip()
    logger.info(f"Rendering {object_name}...")
    try:
        process = subprocess.run(["bash", "-c", command], stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("error running command: ", command)


if __name__ == "__main__":
    fire.Fire(render_objects)
