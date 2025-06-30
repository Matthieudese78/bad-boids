"""An amelioration of the deliberately bad implementation of [Boids](http://dl.acm.org/citation.cfm?doid=37401.37406)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

BIRDS_POPULATION = 50

# %%
# Deliberately terrible code for teaching purposes

boids_x = [np.random.uniform(-450, 50.0) for x in range(BIRDS_POPULATION)]
boids_y = [np.random.uniform(300.0, 600.0) for x in range(BIRDS_POPULATION)]
boid_x_velocities = [np.random.uniform(0, 10.0) for x in range(BIRDS_POPULATION)]
boid_y_velocities = [np.random.uniform(-20.0, 20.0) for x in range(BIRDS_POPULATION)]
boids = (boids_x, boids_y, boid_x_velocities, boid_y_velocities)


def update_boids(boids: tuple):
    xs, ys, xvs, yvs = boids
    delta_xvs = np.zeros_like(xs)
    delta_yvs = np.zeros_like(xs)

    # Fly towards the middle
    for i in range(len(xs)):
        for j in range(len(xs)):
            delta_xvs[i] = delta_xvs[i] + (xs[j] - xs[i]) * 0.01 / len(xs)
    for i in range(len(xs)):
        for j in range(len(xs)):
            delta_yvs[i] = delta_yvs[i] + (ys[j] - ys[i]) * 0.01 / len(xs)
    # Fly away from nearby boids
    for i in range(len(xs)):
        for j in range(len(xs)):
            if (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2 < 100:
                delta_xvs[i] = delta_xvs[i] + (xs[i] - xs[j])
                delta_yvs[i] = delta_yvs[i] + (ys[i] - ys[j])
    # Try to match speed with nearby boids
    for i in range(len(xs)):
        for j in range(len(xs)):
            if (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2 < 10000:
                delta_xvs[i] = delta_xvs[i] + (xvs[j] - xvs[i]) * 0.125 / len(xs)
                delta_yvs[i] = delta_yvs[i] + (yvs[j] - yvs[i]) * 0.125 / len(xs)
    # Update velocities
    for i in range(len(xs)):
        xvs[i] = xvs[i] + delta_xvs[i]
        yvs[i] = yvs[i] + delta_yvs[i]
    # Move according to velocities
    for i in range(len(xs)):
        xs[i] = xs[i] + xvs[i]
        ys[i] = ys[i] + yvs[i]


figure = plt.figure()
axes = plt.axes(xlim=(-500, 1500), ylim=(-500, 1500))
scatter = axes.scatter(boids[0], boids[1])


def ANIMATE(frame):
    update_boids(boids)
    scatter.set_offsets(list(zip(boids[0], boids[1])))


anim = animation.FuncAnimation(figure, ANIMATE, frames=50, interval=50)

if __name__ == "__main__":
    plt.show()

# %%
