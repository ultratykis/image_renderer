"""Implement the Renderer class with blender."""

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

# import bpy first
import bpy
import addon_utils

from .MetadataExtractor import MetadataExtractor
from .utlis import get_scene_root_objects, scene_bbox

addon_utils.enable("measureit")
import numpy as np
from mathutils import Vector


class Renderer:
    """A class to render objects with blender."""

    def __init__(
        self,
        engine: str = "CYCLES",
        freestyle: bool = True,
        visible_edges: bool = False,
        render_size: int = 1024,
        res_percentage: int = 100,
        output_channels: str = "RGBA",
    ) -> None:
        self.bpy = bpy
        self.context = self.bpy.context

        """Initialize the renderer."""
        self.context.scene.render.engine = engine
        self.context.scene.render.resolution_x = render_size
        self.context.scene.render.resolution_y = render_size
        self.context.scene.render.image_settings.file_format = "PNG"
        self.context.scene.render.image_settings.color_mode = output_channels
        self.context.scene.render.resolution_percentage = res_percentage
        self.context.scene.render.film_transparent = True

        # set cycles settings
        self.context.scene.cycles.samples = 128
        self.context.scene.cycles.device = "GPU"
        self.context.scene.render.film_transparent = True

        self.context.preferences.addons["cycles"].preferences.get_devices()
        self.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = (
            "CUDA"  # One in ('NONE', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI')
        )

        if freestyle:
            # set freestyle
            self.context.scene.render.use_freestyle = True
            linesets = self.context.view_layer.freestyle_settings.linesets.active
            linesets.select_contour = True
            linesets.select_crease = True
            linesets.select_edge_mark = True
            linesets.select_external_contour = True
            linesets.select_material_boundary = True
            linesets.select_ridge_valley = False
            linesets.select_silhouette = True
            linesets.select_suggestive_contour = False
            if output_channels == "RGB":
                linesets.linestyle.color = (1, 1, 1)  # set color to white
            # if unvisible is True, render the invisible edges as well and render as dashed lines
            if not visible_edges:
                hidden_lineset = (
                    self.context.view_layer.freestyle_settings.linesets.new(
                        "hidden_lines"
                    )
                )  # add new lineset
                hidden_lineset.visibility = "HIDDEN"
                hidden_lineset.select_contour = True
                hidden_lineset.select_crease = True
                hidden_lineset.select_edge_mark = True
                hidden_lineset.select_external_contour = True
                hidden_lineset.select_material_boundary = True
                hidden_lineset.select_ridge_valley = False
                hidden_lineset.select_silhouette = True
                hidden_lineset.select_suggestive_contour = False
                hidden_lineset.linestyle.use_dashed_line = True  # set to dashed lines
                hidden_lineset.linestyle.color = (0.5, 0.5, 0.5)  # set color to gray
                hidden_lineset.linestyle.dash1 = 5  # set the dash length to 0.1
                hidden_lineset.linestyle.gap1 = 5  # set the gap length to 0.1

            # only output the freestyle lines
            self.context.view_layer.use_solid = False
            # set background to white
            # scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
            # scene.render.film_transparent = False

        # define io functions
        self.IMPORT_FUNCTIONS: Dict[str, Callable] = {
            "obj": bpy.ops.import_scene.obj,
            "glb": bpy.ops.import_scene.gltf,
            "gltf": bpy.ops.import_scene.gltf,
            "usd": bpy.ops.import_scene.usd,
            "fbx": bpy.ops.import_scene.fbx,
            "stl": bpy.ops.import_mesh.stl,
            "usda": bpy.ops.import_scene.usda,
            "dae": bpy.ops.wm.collada_import,
            "ply": bpy.ops.import_mesh.ply,
            "abc": bpy.ops.wm.alembic_import,
            "blend": bpy.ops.wm.append,
        }

    def render_object(
        self,
        object_file: Path,
        output_dir: Path,
        only_northern_hemisphere: bool = False,
        camera_type: str = "ORTHO",
        num_renders: int = 12,
        three_views: bool = False,
        freestyle: bool = True,
        camera: List[float] = None,
        error_az_range=22.5,
        error_el_range=5,
        camera_dist=1.5,
    ) -> None:
        """Saves rendered images with its camera matrix and metadata of the object.

        Args:
            object_file (Path): Path to the object file.
            output_dir (Path): Path to the directory where the rendered images and metadata
                will be saved.
            only_northern_hemisphere (bool): Whether to only render sides of the object that
                are in the northern hemisphere. This is useful for rendering objects that
                are photogrammetrically scanned, as the bottom of the object often has
                holes.
            camera_type (str): Type of camera to use. Must be one of "PERSP", "ORTHO".
            three_views (bool): Whether to render the object from 3 views (front, side, and top). If true, num_renders and num_trials are ignored.
            num_renders (int): Number of renders to save of the object.
            num_trials (int): Number of trials to try rendering num_renders images.
            freestyle (bool, optional): Whether to render the object with freestyle. Defaults to False.
            error_az_range (float, optional): Range of error in azimuth angle. Defaults to 22.5.
            error_el_range (float, optional): Range of error in elevation angle. Defaults to 5.
            camera_dist (float, optional): Distance of the camera from the object. Defaults to 1.5.
        Returns:
            None
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir.joinpath("metadata.json")

        # load the object
        if object_file.suffix == ".blend":
            bpy.ops.object.mode_set(mode="OBJECT")
            self.reset_cameras(camera_type=camera_type)
            self.delete_invisible_objects()
        else:
            self.reset_scene()
            self.load_object(object_file)

        # normalize the scene
        self.normalize_scene()

        # randomize the lighting
        self.randomize_lighting()

        # Set up cameras
        cam = self.context.scene.objects["Camera"]
        cam.data.lens = 35
        cam.data.sensor_width = 32

        # Set up camera constraints
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        empty = bpy.data.objects.new("Empty", None)
        self.context.scene.collection.objects.link(empty)
        cam_constraint.target = empty

        # Extract the metadata. This must be done before normalizing the scene to get
        # accurate bounding box information.
        metadata_extractor = MetadataExtractor(
            object_path=object_file, scene=self.context.scene, bdata=bpy.data
        )
        metadata = metadata_extractor.get_metadata()

        # delete all objects that are not meshes
        if object_file.suffix == ".usdz":
            # don't delete missing textures on usdz files, lots of them are embedded
            missing_textures = None
        else:
            missing_textures = self.delete_missing_textures()
        metadata["missing_textures"] = missing_textures

        # possibly apply a random color to all objects
        if (
            object_file.suffix == ".stl" or object_file.suffix == ".ply"
        ) and not freestyle:
            # bpy.ops.object.select_by_type(type="MESH")
            # assert len(bpy.context.selected_objects) == 1
            rand_color = self.apply_single_random_color_to_all_objects()
            metadata["random_color"] = rand_color
            # bpy.ops.object.select_all(action="DESELECT")
        else:
            metadata["random_color"] = None
        if camera is None:
            if three_views:
                num_renders = 3
                preset_cameras = [
                    {"az": 0, "el": 0},
                    {"az": 90, "el": 0},
                    {"az": 0, "el": 90},
                ]

            az_angles = []
            el_angles = []
            for view_id in range(num_renders):
                if not three_views:
                    # set the camera position
                    # randomly perturb the camera position from the 0 to 360 degrees azimuth.
                    # sampled by 360 / num_renders. azimuth degree increases each view_id
                    az = (
                        view_id * 360 / num_renders
                        + (2 * np.random.rand() - 1) * error_az_range
                    )
                    el = (2 * np.random.rand() - 1) * error_el_range + 20
                else:
                    az = preset_cameras[view_id]["az"]
                    el = preset_cameras[view_id]["el"]
                az_angles.append(az)
                el_angles.append(el)
                cam.location = (
                    camera_dist
                    * math.cos(el / 180 * math.pi)
                    * math.sin(az / 180 * math.pi),
                    camera_dist
                    * math.cos(el / 180 * math.pi)
                    * math.cos(az / 180 * math.pi),
                    camera_dist * math.sin(el / 180 * math.pi)
                    if not only_northern_hemisphere
                    else abs(camera_dist * math.sin(el / 180 * math.pi)),
                )

                direction = Vector((0, 0, 0)) - cam.location
                rot_quat = direction.to_track_quat("-Z", "Y")
                cam.rotation_euler = rot_quat.to_euler()
                # render the image
                filename = output_dir.joinpath(f"{view_id}.png")
                self.context.scene.render.filepath = str(filename)
                bpy.ops.render.render(write_still=True)

            metadata["camera"] = {"az": az_angles, "el": el_angles}
            # save metadata
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, sort_keys=True, indent=2)
        else:
            az, el, camera_dist = camera
            cam.location = (
                camera_dist
                * math.cos(el / 180 * math.pi)
                * math.sin(az / 180 * math.pi),
                camera_dist
                * math.cos(el / 180 * math.pi)
                * math.cos(az / 180 * math.pi),
                camera_dist * math.sin(el / 180 * math.pi)
                if not only_northern_hemisphere
                else abs(camera_dist * math.sin(el / 180 * math.pi)),
            )

            direction = Vector((0, 0, 0)) - cam.location
            rot_quat = direction.to_track_quat("-Z", "Y")
            cam.rotation_euler = rot_quat.to_euler()
            # render the image
            filename = output_dir.joinpath(f"rendered.png")
            self.context.scene.render.filepath = str(filename)
            bpy.ops.render.render(write_still=True)

    def reset_cameras(self, camera_type: str = "PERSP") -> None:
        """Resets the cameras in the scene to a single default camera.

        Args:
            camera_type (str): Type of camera to use. Must be one of "PERSP", "ORTHO".
        Returns:
            None
        """
        # Delete all existing cameras
        self.bpy.ops.object.select_all(action="DESELECT")
        self.bpy.ops.object.select_by_type(type="CAMERA")
        self.bpy.ops.object.delete()

        # Create a new camera with default properties
        self.bpy.ops.object.camera_add()

        # Rename the new camera to 'NewDefaultCamera'
        new_camera = self.bpy.context.active_object
        new_camera.name = "Camera"
        # Set the camera type
        new_camera.type = camera_type
        if camera_type == "ORTHO":
            new_camera.ortho_scale = 1.5

        # Set the new camera as the active camera for the scene
        self.context.scene.camera = new_camera

    def _sample_spherical(
        self,
        radius_min: float = 1.5,
        radius_max: float = 2.0,
        maxz: float = 1.6,
        minz: float = -0.75,
    ) -> np.ndarray:
        """Sample a random point in a spherical shell.

        Args:
            radius_min (float): Minimum radius of the spherical shell.
            radius_max (float): Maximum radius of the spherical shell.
            maxz (float): Maximum z value of the spherical shell.
            minz (float): Minimum z value of the spherical shell.

        Returns:
            np.ndarray: A random (x, y, z) point in the spherical shell.
        """
        correct = False
        vec = np.array([0, 0, 0])
        while not correct:
            vec = np.random.uniform(-1, 1, 3)
            #         vec[2] = np.abs(vec[2])
            radius = np.random.uniform(radius_min, radius_max, 1)
            vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
            if maxz > vec[2] > minz:
                correct = True
        return vec

    def randomize_camera(
        self,
        radius_min: float = 1.5,
        radius_max: float = 2.2,
        maxz: float = 2.2,
        minz: float = -2.2,
        only_northern_hemisphere: bool = False,
    ) -> bpy.types.Object:
        """Randomizes the camera location and rotation inside of a spherical shell.

        Args:
            radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
                1.5.
            radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
                2.0.
            maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
            minz (float, optional): Minimum z value of the spherical shell. Defaults to
                -0.75.
            only_northern_hemisphere (bool, optional): Whether to only sample points in the
                northern hemisphere. Defaults to False.

        Returns:
            bpy.types.Object: The camera object.
        """

        x, y, z = self._sample_spherical(
            radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
        )
        camera = bpy.data.objects["Camera"]

        # only positive z
        if only_northern_hemisphere:
            z = abs(z)

        camera.location = np.array([x, y, z])

        direction = -camera.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera.rotation_euler = rot_quat.to_euler()

        return camera

    def _create_light(
        self,
        name: str,
        light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
        location: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        energy: float,
        use_shadow: bool = False,
        specular_factor: float = 1.0,
    ):
        """Creates a light object.

        Args:
            name (str): Name of the light object.
            light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
            location (Tuple[float, float, float]): Location of the light.
            rotation (Tuple[float, float, float]): Rotation of the light.
            energy (float): Energy of the light.
            use_shadow (bool, optional): Whether to use shadows. Defaults to False.
            specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

        Returns:
            bpy.types.Object: The light object.
        """

        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_object = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = location
        light_object.rotation_euler = rotation
        light_data.use_shadow = use_shadow
        light_data.specular_factor = specular_factor
        light_data.energy = energy
        return light_object

    def randomize_lighting(self) -> Dict[str, bpy.types.Object]:
        """Randomizes the lighting in the scene.

        Returns:
            Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
                "key_light", "fill_light", "rim_light", and "bottom_light".
        """

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        # Create key light
        key_light = self._create_light(
            name="Key_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, -0.785398),
            energy=random.choice([3, 4, 5]),
        )

        # Create fill light
        fill_light = self._create_light(
            name="Fill_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, 2.35619),
            energy=random.choice([2, 3, 4]),
        )

        # Create rim light
        rim_light = self._create_light(
            name="Rim_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(-0.785398, 0, -3.92699),
            energy=random.choice([3, 4, 5]),
        )

        # Create bottom light
        bottom_light = self._create_light(
            name="Bottom_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(3.14159, 0, 0),
            energy=random.choice([1, 2, 3]),
        )

        return dict(
            key_light=key_light,
            fill_light=fill_light,
            rim_light=rim_light,
            bottom_light=bottom_light,
        )

    def reset_scene(self) -> None:
        """Resets the scene to a clean state.

        Returns:
            None
        """
        # delete everything that isn't part of a camera or a light
        for obj in bpy.data.objects:
            if obj.type not in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)

        # delete all the materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)

        # delete all the textures
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)

        # delete all the images
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

    def load_object(self, object_path: Path) -> None:
        """Loads a model with a supported file extension into the scene.

        Args:
            object_path (Path): Path to the model file.

        Raises:
            ValueError: If the file extension is not supported.

        Returns:
            None
        """
        file_extension = object_path.suffix.lower()[1:]
        if file_extension is None or file_extension == "usdz":
            raise ValueError(f"Unsupported file type: {object_path}")

        # load from existing import functions
        import_function = self.IMPORT_FUNCTIONS[file_extension]

        if file_extension == "blend":
            import_function(directory=str(object_path), link=False)
        elif file_extension in {"glb", "gltf"}:
            import_function(filepath=str(object_path), merge_vertices=True)
        else:
            import_function(filepath=str(object_path))

    def delete_invisible_objects(
        self,
    ) -> None:
        """Deletes all invisible objects in the scene.

        Returns:
            None
        """
        bpy.ops.object.select_all(action="DESELECT")
        for obj in self.context.scene.objects:
            if obj.hide_viewport or obj.hide_render:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.hide_select = False
                obj.select_set(True)
        bpy.ops.object.delete()

        # Delete invisible collections
        invisible_collections = [
            col for col in bpy.data.collections if col.hide_viewport
        ]
        for col in invisible_collections:
            bpy.data.collections.remove(col)

    def normalize_scene(
        self,
    ) -> None:
        """Normalizes the scene by scaling and translating it to fit in a unit cube centered
        at the origin.

        Mostly taken from the Point-E / Shap-E rendering script
        (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
        but fix for multiple root objects: (see bug report here:
        https://github.com/openai/shap-e/pull/60).

        Returns:
            None
        """
        if len(list(get_scene_root_objects())) > 1:
            # create an empty object to be used as a parent for all root objects
            parent_empty = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(parent_empty)

            # parent all root objects to the empty object
            for obj in get_scene_root_objects():
                if obj != parent_empty:
                    obj.parent = parent_empty

        bbox_min, bbox_max = scene_bbox()
        scale = 1 / max(bbox_max - bbox_min)
        for obj in get_scene_root_objects():
            obj.scale = obj.scale * scale

        # Apply scale to matrix_world.
        bpy.context.view_layer.update()
        bbox_min, bbox_max = scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        for obj in get_scene_root_objects():
            obj.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")

        # unparent the camera
        bpy.data.objects["Camera"].parent = None

    def delete_missing_textures(
        self,
    ) -> Dict[str, Any]:
        """Deletes all missing textures in the scene.

        Returns:
            Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
                "count" is the number of missing textures, "files" is a list of the missing
                texture file paths, and "file_path_to_color" is a dictionary mapping the
                missing texture file paths to a random color.
        """
        missing_file_count = 0
        out_files = []
        file_path_to_color = {}

        # Check all materials in the scene
        for material in bpy.data.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            file_path = bpy.path.abspath(image.filepath)
                            if file_path == "":
                                # means it's embedded
                                continue

                            if not os.path.exists(file_path):
                                # Find the connected Principled BSDF node
                                connected_node = node.outputs[0].links[0].to_node

                                if connected_node.type == "BSDF_PRINCIPLED":
                                    if file_path not in file_path_to_color:
                                        # Set a random color for the unique missing file path
                                        random_color = [
                                            random.random() for _ in range(3)
                                        ]
                                        file_path_to_color[file_path] = random_color + [
                                            1
                                        ]

                                    connected_node.inputs[
                                        "Base Color"
                                    ].default_value = file_path_to_color[file_path]

                                # Delete the TEX_IMAGE node
                                material.node_tree.nodes.remove(node)
                                missing_file_count += 1
                                out_files.append(image.filepath)
        return {
            "count": missing_file_count,
            "files": out_files,
            "file_path_to_color": file_path_to_color,
        }

    def _get_random_color(
        self,
    ) -> Tuple[float, float, float, float]:
        """Generates a random RGB-A color.

        The alpha value is always 1.

        Returns:
            Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
            range [0, 1].
        """
        return (random.random(), random.random(), random.random(), 1)

    def _apply_color_to_object(
        self, obj: bpy.types.Object, color: Tuple[float, float, float, float]
    ) -> None:
        """Applies the given color to the object.

        Args:
            obj (bpy.types.Object): The object to apply the color to.
            color (Tuple[float, float, float, float]): The color to apply to the object.

        Returns:
            None
        """
        mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        principled_bsdf = nodes.get("Principled BSDF")
        if principled_bsdf:
            principled_bsdf.inputs["Base Color"].default_value = color
        obj.data.materials.append(mat)

    def apply_single_random_color_to_all_objects(
        self,
    ) -> Tuple[float, float, float, float]:
        """Applies a single random color to all objects in the scene.

        Returns:
            Tuple[float, float, float, float]: The random color that was applied to all
            objects.
        """
        rand_color = self._get_random_color()
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                self._apply_color_to_object(obj, rand_color)
        return rand_color

    def gc_collect(self) -> None:
        """Runs garbage collection.

        Returns:
            None
        """
        import gc

        del self.bpy
        gc.collect()
