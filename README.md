# Render 3D model to multi-view images (or contours/sketches) with blender.

## Usage:

```bash
pip install bpy numpy

python render.py --object_path <path/to/file> --output_dir <path/to/output/dir> --engine CYCLES --num_renders 12 --num_trials 1
```

Add `--freestyle` flag to render contours/sketches.

Add `--only_northern_hemisphere` flag to locate camera only northern hemisphere.

This program will render `num_renders × num_trials` images. While `num_trials` not `1`, the camera will be placed at `0, 45, 90, 135, 180` degrees azimuth and randomly perturbed to render num_renders images. Otherwise, the camera will be randomly placed from the 0 to 360 degrees azimuth.

Code heavily borrowed from [Objaverse-XL](https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering).
