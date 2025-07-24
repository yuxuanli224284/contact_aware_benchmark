# Contact-aware VLA benchmark

## Create new environments (.bddl files)
1. Environment with spatial semantics:
Create a custom environment with clusters of different density levels.

``bash
python create_spatial_bddl.py
``

2. Environment with property semantics:
Create a custom environment with items of different danger level.

``bash
python create_property_bddl.py
``

3. Environment with object combination:
Create a custom environment with customizable object combinations

``bash
python create_comb_bddl.py
``

4. Environment with immovable objects:
Create a custom environment with immovable objects:

``bash
python create_immovable_bddl.py
``