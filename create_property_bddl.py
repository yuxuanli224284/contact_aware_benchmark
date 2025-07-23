import argparse
import numpy as np
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info

# Danger coefficient table
DANGER_OBJECTS = {
    "alphabet_soup": 7,
    "ketchup": 6,
    "butter": 3,
    "milk": 8,
    "porcelain_mug": 6,
    "yellow_book": 2,
    "plate": 6,
    "akita_black_bowl": 6
}

class PropertySceneDynamic(InitialSceneTemplates):
    def __init__(self, target_object, target_position, danger_coeff_list, only_front=True, distractor_radius_range=(0.1, 0.25), workspace_name="kitchen_table"):
        self.workspace_name = workspace_name
        self.target_object = target_object
        self.target_position = target_position
        self.danger_coeff_list = danger_coeff_list
        self.only_front = only_front
        self.r_min, self.r_max = distractor_radius_range

        # Choose distractor objects from danger coefficients
        danger_to_objects = {}
        for obj, coeff in DANGER_OBJECTS.items():
            danger_to_objects.setdefault(coeff, []).append(obj)

        self.distractor_objects = []
        for coeff in danger_coeff_list:
            candidates = danger_to_objects.get(coeff, [])
            if not candidates:
                raise ValueError(f"No object found with danger coefficient {coeff}")
            choice = np.random.choice(candidates)
            self.distractor_objects.append(choice)

        print(f"[INFO] Chosen distractor objects (based on danger_coeff_list={danger_coeff_list}): {self.distractor_objects}")

        fixture_num_info = {workspace_name: 1}
        object_num_info = {target_object: 1}
        for obj in self.distractor_objects:
            object_num_info[obj] = object_num_info.get(obj, 0) + 1

        super().__init__(workspace_name, fixture_num_info, object_num_info)

    def define_regions(self):
        self.regions.update(self.get_region_dict(
            region_centroid_xy=self.target_position,
            region_name=f"{self.target_object}_init_region",
            target_name=self.workspace_name,
            region_half_len=0.025
        ))

        for i, obj in enumerate(self.distractor_objects):
            max_attempts = 100
            for _ in range(max_attempts):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(self.r_min, self.r_max)
                offset = radius * np.array([np.cos(angle), np.sin(angle)])
                pos = np.array(self.target_position) + offset
                if not self.only_front or pos[0] < self.target_position[0]:
                    break

            pos = pos.tolist()
            region_name = f"{obj}_{i}_region"
            self.regions.update(self.get_region_dict(
                region_centroid_xy=pos,
                region_name=region_name,
                target_name=self.workspace_name,
                region_half_len=0.025
            ))

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [("On", f"{self.target_object}_1", f"{self.workspace_name}_{self.target_object}_init_region")]
        for i, obj in enumerate(self.distractor_objects):
            region_name = f"{obj}_{i}_region"
            states.append(("On", f"{obj}_1", f"{self.workspace_name}_{region_name}"))
        return states

def create_and_register_scene(args):
    @register_mu(scene_type="property")
    class PropertyScene(PropertySceneDynamic):
        def __init__(self):
            super().__init__(
                target_object=args.target_object,
                target_position=args.target_position,
                danger_coeff_list=args.danger_coeff_list,
                only_front=args.only_front,
                distractor_radius_range=args.distractor_radius_range
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="property_scene")
    parser.add_argument("--goal_object", type=str, default="akita_black_bowl_1")
    parser.add_argument("--goal_region", type=str, default="kitchen_table_white_bowl_init_region")
    parser.add_argument("--folder", type=str, default="libero/libero/bddl_files/libero_spatial")
    parser.add_argument("--task_language", type=str, default="Put the white bowl into the cabinet")

    parser.add_argument("--target_object", type=str, default="akita_black_bowl")
    parser.add_argument("--target_position", type=tuple, default=(0.0, 0.0))
    parser.add_argument("--danger_coeff_list", type=list, default=[8, 7, 3, 2, 6])
    parser.add_argument("--only_front", type=bool, default=False)
    parser.add_argument("--distractor_radius_range", type=tuple, default=(0.05, 0.15))

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
