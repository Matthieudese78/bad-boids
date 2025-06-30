# %%
"""An amelioration of the deliberately bad implementation of [Boids](http://dl.acm.org/citation.cfm?doid=37401.37406)."""

from future import annotations
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

BIRDS_POPULATION = 50
REPELLING_DISTANCE = 100.0
ATTRACTING_DISTANCE = 10000.0
ATTRACTION_RATE = 0.125
MIDDLE_ATTRACTION_RATE = 0.01

# %%
# Deliberately terrible code for teaching purposes

boids_x = [np.random.uniform(-450, 50.0) for x in range(BIRDS_POPULATION)]
boids_y = [np.random.uniform(300.0, 600.0) for x in range(BIRDS_POPULATION)]
boid_x_velocities = [np.random.uniform(0, 10.0) for x in range(BIRDS_POPULATION)]
boid_y_velocities = [np.random.uniform(-20.0, 20.0) for x in range(BIRDS_POPULATION)]
boids = (boids_x, boids_y, boid_x_velocities, boid_y_velocities)
print(np.shape(boids))

position = np.array([boids[0], boids[1]]).T
velocity = np.array([boids[1], boids[2]]).T


# %%
class Boids:
    """A class to represent the boids simulation."""

    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity

    def update_boids(self) -> None:
        """Update the positions and velocities of the boids."""
        position = self.position
        velocity = self.velocity
        delta_velocity = np.zeros_like(velocity)

        for i, j in product(range(BIRDS_POPULATION), repeat=2):
            # Fly towards the middle
            delta_velocity[i] = (
                delta_velocity[i] + (position[j] - position[i]) * MIDDLE_ATTRACTION_RATE / BIRDS_POPULATION
            )

            # Fly away from nearby boids
            if np.linalg.norm(position[j] - position[i]) ** 2 < REPELLING_DISTANCE:
                delta_velocity[i] = delta_velocity[i] + (position[i] - position[j])

            # Try to match speed with nearby boids
            if np.linalg.norm(position[j] - position[i]) ** 2 < ATTRACTING_DISTANCE:
                delta_velocity[i] = delta_velocity[i] + (velocity[j] - velocity[i]) * ATTRACTION_RATE / BIRDS_POPULATION

        # Update positions & velocities
        self.velocity = self.velocity + delta_velocity
        self.position = self.position + velocity


def animate(boids: Boids) -> None:
    boids.update_boids()
    scatter.set_offsets(list(zip(boids.position[:, 0], boids.position[:, 1])))


if __name__ == "__main__":
    boids = Boids(position, velocity)
    figure = plt.figure()
    axes = plt.axes(xlim=(-500, 1500), ylim=(-500, 1500))
    scatter = axes.scatter(boids.position[:, 0], boids.position[:, 1])

    anim = animation.FuncAnimation(figure, animate, frames=50, blit=True)
    plt.show()

# %%
