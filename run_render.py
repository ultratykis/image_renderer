import glob
import os

import fire
from loguru import logger
from multiprocessing import Pool
import subprocess


def render_objects(
    raw_data_path: str = "sample_data",
    render_output_dir: str = "sample_output",
    num_renders: int = 5,
    num_trials: int = 5,
    freestyle: bool = True,
    engine: str = "CYCLES",
    only_northern_hemisphere: bool = False,
    render_size: int = 512,
    res_percentage: int = 100,
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
    all_objects = glob.glob("*.stl", root_dir=raw_data_path, recursive=True)
    logger.info(f"Found {len(all_objects)} objects to render.")

    # set render style
    if freestyle:
        args += " --freestyle"
    args += f" --engine {engine}"
    args += f" --render_size {render_size}"
    args += f" --res_percentage {res_percentage}"
    args += f" --num_renders {num_renders}"
    args += f" --num_trials {num_trials}"
    command = f"python blender.py {args}"

    # get (input, output) paths
    io_paths = [
        (
            os.path.join(raw_data_path, os.path.basename(obj)),
            os.path.join(render_output_dir, os.path.basename(obj).split(".")[0]),
        )
        for obj in all_objects
    ]
    commands = [
        command + f" --object_path {obj} --output_dir {output_dir}"
        for obj, output_dir in io_paths
    ]

    # render all objects in parallel
    # get cpu count
    cpu_count = os.cpu_count()
    with Pool(processes=cpu_count) as p:
        p.map(subprocess_cmd, commands)

    print("Done rendering all objects.")


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
