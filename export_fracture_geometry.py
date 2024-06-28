import numpy as np

import model_setup
from geometries import RandomFractures3d


class Model(RandomFractures3d, model_setup.physical_models()[0][1]):
    pass


if __name__ == "__main__":
    # Parameters
    params = {"fracture_size": 0.25, "num_fractures": 8}
    model = Model(params)
    model.set_materials()
    model.set_fractures()
    points = np.vstack([f.pts for f in model.fractures])
    header = "Each set of three rows represents a fracture's x, y and z coordinates. "
    header += "Each column represents one of the fractures' eight vertices."
    np.savetxt("fracture_points.csv", points, delimiter=",", header=header)
