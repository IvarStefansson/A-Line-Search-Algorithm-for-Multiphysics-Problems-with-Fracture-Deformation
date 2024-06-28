import porepy as pp

import material_constant_values as materials
from line_search import NonlinearSolver

solid = pp.SolidConstants(materials.plain_rock_values)
fluid = pp.FluidConstants(materials.fluid_values)
params = {
    "max_iterations": 100,
    "linear_solver": "scipy_sparse",  # Don't trust pypardiso for this problem.
    "nl_convergence_tol": 1e-5,
    "nl_convergence_tol_res": 1e-10,
    "export_constants_separately": False,
    "material_constants": {"solid": solid, "fluid": fluid},
    "fracture_indices": [-1],  # Has constant -1-coordinate (2d y, 3d z).
    "nonlinear_solver": NonlinearSolver,
    "meshing_arguments": {"cell_size": 1 / 10, "cell_size_fracture": 1e-1},
}
algorithms = [
    {"Local_line_search": 0},
    {"Global_line_search": 1},
    {"Local_line_search": 1},
    {"Local_line_search": 1, "relative_violation": 1},
]
algorithm_names = [
    "No LS",
    "RLS",
    r"CLS $i_c$",
    r"CLS $i_a$",
]
