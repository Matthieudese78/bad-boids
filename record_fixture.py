# %%
from copy import deepcopy

import yaml

from boids.boids import boids, update_boids
import numpy as np

# %%
print(np.shape(boids))

before = deepcopy(boids)
after = update_boids(before)
fixture = {"before": before, "after": after}
with open("fixture.yml", "w") as fixture_file:
    fixture_file.write(yaml.safe_dump(fixture))

# %%
