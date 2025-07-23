import argparse
import numpy as np
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info

VALID_IMMOVABLE_OBJECTS = ["white_cabinet", "wooden_shelf", "short_fridge"]

class CombSceneDynamic(InitialSceneTemplates):
    def __init__(self, target_object, target_position, region_range, object_pairs, workspace_name="kitchen_table"):
        self.workspace_name = workspace_name
        self.target_object = target_object
        self.target_position = target_position
        self.region_range = region_range
        self.object_pairs = object_pairs

        fixture_num_info = {workspace_name: 1}
        object_num_info = {target_object: 1}

        self.custom_immovable_objects = []
        self.custom_movable_objects = []
        self.on_top_mapping = {}  # Store which object is on top of which

        for immovable, movable in object_pairs:
            if immovable != "none":
                self.custom_immovable_objects.append(immovable)
                fixture_num_info[f"{immovable}"] = 1  # Treat immovable as fixture
            if movable != "none":
                object_num_info[movable] = object_num_info.get(movable, 0) + 1
                self.custom_movable_objects.append(movable)
            if immovable != "none" and movable != "none":
                self.on_top_mapping[movable] = immovable

        super().__init__(workspace_name, fixture_num_info, object_num_info)

    def define_regions(self):
        self.regions.update(self.get_region_dict(
            region_centroid_xy=self.target_position,
            region_name=f"{self.target_object}_init_region",
            target_name=self.workspace_name,
            region_half_len=0.025
        ))

        r_min, r_max = self.region_range
        for i, (immovable, movable) in enumerate(self.object_pairs):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(r_min, r_max)
            offset = radius * np.array([np.cos(angle), np.sin(angle)])
            base_pos = self.target_position + offset

            if immovable != "none":
                region_name = f"{immovable}_1_region"
                self.regions.update(self.get_region_dict(
                    region_centroid_xy=base_pos.tolist(),
                    region_name=region_name,
                    target_name=self.workspace_name,
                    region_half_len=0.03
                ))

            if movable != "none" and immovable == "none":
                region_name = f"{movable}_1_region"
                self.regions.update(self.get_region_dict(
                    region_centroid_xy=base_pos.tolist(),
                    region_name=region_name,
                    target_name=self.workspace_name,
                    region_half_len=0.025
                ))

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", f"{self.target_object}_1", f"{self.workspace_name}_{self.target_object}_init_region")
        ]

        for immovable, movable in self.object_pairs:
            if immovable != "none":
                region_name = f"{immovable}_1_region"
                states.append(("On", f"{immovable}_1", f"{self.workspace_name}_{region_name}"))
            if movable != "none":
                if immovable != "none":
                    states.append(("On", f"{movable}_1", f"{immovable}_1_top_region"))
                else:
                    region_name = f"{movable}_1_region"
                    states.append(("On", f"{movable}_1", f"{self.workspace_name}_{region_name}"))

        return states

def create_and_register_scene(args):
    @register_mu(scene_type="comb")
    class CombScene(CombSceneDynamic):
        def __init__(self):
            super().__init__(
                target_object=args.target_object,
                target_position=args.target_position,
                region_range=args.region_range,
                object_pairs=args.object_pairs
            )
    return CombScene

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="comb_scene")
    parser.add_argument("--goal_object", type=str, default="akita_black_bowl_1")
    parser.add_argument("--goal_region", type=str, default="kitchen_table_akita_black_bowl_init_region")
    parser.add_argument("--folder", type=str, default="libero/libero/bddl_files/libero_spatial")
    parser.add_argument("--task_language", type=str, default="Put the white bowl into the cabinet")

    parser.add_argument("--target_object", type=str, default="akita_black_bowl")
    parser.add_argument("--target_position", type=tuple, default=(0.0, 0.0))
    parser.add_argument("--region_range", type=tuple, default=(0.10, 0.30))
    parser.add_argument("--object_pairs", type=eval, default=[("white_cabinet", "milk"), ("none", "ketchup"), ("wooden_shelf", "none")])

    args = parser.parse_args()

    create_and_register_scene(args)

    register_task_info(
        language=args.task_language,
        scene_name=args.scene_name,
        objects_of_interest=[args.goal_object],
        goal_states=[("On", args.goal_object, args.goal_region)]
    )

    bddl_file_names, failures = generate_bddl_from_task_info(folder=args.folder)
    print("Generated:", bddl_file_names)
    print("Failures:", failures)
