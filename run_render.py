import glob
import os
import subprocess
from multiprocessing import Pool
from typing import List

import fire
import numpy as np
from loguru import logger


def render_object(
    filename: str,
    output_dir: str = None,
    three_views: bool = False,
    num_renders: int = 5,
    num_trials: int = 5,
    freestyle: bool = False,
    visible_edges: bool = True,
    engine: str = "CYCLES",
    only_northern_hemisphere: bool = False,
    render_size: int = 1024,
    res_percentage: int = 100,
    output_channels: str = "RGBA",
) -> bool:
    """Render the provided object and write to 'output_dir' or return as a np.array.

    Args:
        filename: Path to the object to render.
        output_dir: Path to the output directory.
        three_views: Whether to render three views of the object.
        num_renders: Number of views to render per object.
        num_trials: Number of trials to render per view.
        freestyle: Whether to render the object with freestyle.
        visible_edges: Whether to render only the visible edges.
        engine: Rendering engine to use. Either CYCLES or EEVEE.
        only_northern_hemisphere: Whether to render only the northern hemisphere.
        render_size: Size of the rendered image.
        res_percentage: Rescale percentage of the render size to use.
        output_channels: Output channels to use. Either RGBA or RGB.

    Returns: True if the object was rendered successfully, False otherwise.
    """
    args = ""
    # check if we should only render the northern hemisphere
    if only_northern_hemisphere:
        args += "--only_northern_hemisphere"

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
    args += f" --object_filename {filename}"
    args += f" --output_dir {output_dir}"
    command = f"python blender.py {args}"
    logger.debug(f"Running command: {command}")

    # run
    fire.Fire(command)


def render_objects(
    object_path: str = "sample_data",
    output_dir: str = "sample_output",
    three_views: bool = False,
    num_renders: int = 5,
    num_trials: int = 5,
    freestyle: bool = False,
    visible_edges: bool = True,
    engine: str = "CYCLES",
    only_northern_hemisphere: bool = False,
    render_size: int = 1024,
    res_percentage: int = 100,
    output_channels: str = "RGBA",
) -> bool:
    """Render the objects in the raw data path.
    Args:
        object_path: Path to the raw data.
        output_dir: Path to the output directory.
        three_views: Whether to render three views of the object.
        num_renders: Number of views to render per object.
        num_trials: Number of trials to render per view.
        freestyle: Whether to render the object with freestyle.
        visible_edges: Whether to render only the visible edges.
        engine: Rendering engine to use. Either CYCLES or EEVEE.
        only_northern_hemisphere: Whether to render only the northern hemisphere.
        render_size: Size of the rendered image.
        res_percentage: Rescale percentage of the render size to use.
        output_channels: Output channels to use. Either RGBA or RGB.

    Returns: True if the object was rendered successfully, False otherwise.
    """
    args = ""
    # check if we should only render the northern hemisphere
    if only_northern_hemisphere:
        args += "--only_northern_hemisphere"

    # get all objects in the raw data path
    all_objects = glob.glob("**/*.STL", root_dir=object_path, recursive=True)
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
        command + f" --object_filename '{obj}' --output_dir '{output_dir}'"
        for obj, output_dir in io_paths
    ]

    # get cpu count
    cpu_count = os.cpu_count()
    p_count = 4
    if len(all_objects) > cpu_count:
        # render all objects in parallel
        with Pool(processes=p_count) as p:
            p.map(subprocess_cmd, commands)
    else:
        # render all objects sequentially
        for command in commands:
            subprocess_cmd(command)

    logger.info(f"Done rendering {len(all_objects)} objects.")


def subprocess_cmd(command):
    # get object name
    object_name = command.split("--object_filename")[1].split("--")[0].strip()
    logger.info(f"Rendering {object_name}...")
    try:
        process = subprocess.run(["bash", "-c", command], stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("error running command: ", command)


if __name__ == "__main__":
    fire.Fire(render_objects)
