"""Run script for application example 1.

We solve mixed-dimensional poroelasticity in a domain with two fractures.

Fracture 1 is less prone to slip than fracture 2 due to higher friction coefficient.
It is also more permeable thanks to higher maximum closure and reference fracture gap.

"""

import logging
import os
from typing import Callable

# Set number of threads to 1
os.environ["NUMBA_NUM_THREADS"] = "1"

import pickle
import time

import numpy as np
import pandas as pd
import porepy as pp
import scipy.sparse as sps

import model_setup
from boundary_conditions import NeumannBoundaryConditions, PureCompression
from common_params import algorithm_names, algorithms, params
from contact_mechanics_utilities import run_and_report_single
from geometries import RandomFractures3d
from visualisation import heatmap

if __name__ == "__main__":
    # Set logging level
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    plot_and_save = True

    h = 0.15
    params.update(
        {
            "meshing_arguments": {"cell_size": h, "cell_size_fracture": h * 0.5},
            "fracture_size": 0.25,
        }
    )

    t_0 = time.time()
    for phys in range(2)[:]:
        phys_name, phys_model = model_setup.physical_models()[phys]
        base_name = f"multiple_fractures/{phys_name}"
        columns = []
        all_iterations = {}
        all_data = {}
        for num_fractures in [4, 8][:]:
            for dilation in [0.1, 0.2][:]:
                subcase_name = f"fractures_{num_fractures}_dilation_{dilation}"
                col_name = f"{num_fractures} fracs\n" + r"$\phi$" + f"={dilation:.1f}"
                columns.append(col_name)
                params["material_constants"]["solid"]._constants[
                    "dilation_angle"
                ] = dilation
                case_name = base_name + "/" + subcase_name
                logger.info(f"\n\nRunning {case_name}")

                params.update(
                    {
                        "time_manager": pp.TimeManager([0, 1e6], 1e6, constant_dt=True),
                        "folder_name": "results/" + case_name,
                        "solver_statistics_file_name": "solver_statistics.json",
                        "num_fractures": num_fractures,
                    },
                )

                class Model(
                    model_setup.FixPrimalVariables,
                    PureCompression,
                    RandomFractures3d,
                    NeumannBoundaryConditions,
                    phys_model,
                ):
                    """Model class"""

                def sort_criterion(cls, domain):
                    return domain.dim == cls.nd

                methods = ["darcy_flux"]
                dofs = [("faces", 1)]

                if phys > 0:
                    methods.append("fourier_flux")
                    dofs.append(("faces", 1))
                for method, dof in zip(methods, dofs):
                    model_setup.override_methods(Model, method, dof, sort_criterion)

                iters = []
                keys = []
                # Loop over combinations of local and global line search values
                for params_loc, name in zip(algorithms, algorithm_names):
                    data_loc, model = run_and_report_single(
                        Model,
                        params,
                        params_loc,
                    )
                    iters.append(data_loc["iterations"])
                    keys.append(name)

                    all_data[subcase_name + name] = data_loc
                all_iterations[col_name] = iters

        if plot_and_save:
            df = pd.DataFrame(all_iterations, index=keys, columns=columns)
            # Create directory for storing results
            heatmap(
                df,
                0,
                params["max_iterations"],
                file_name=base_name + "/iteration_table.png",
                title=phys_name,
                figsize=(5.0, 3.5),
            )
            # Save all data
            with open(base_name + "/all_data.pkl", "wb") as f:
                pickle.dump({"all_data": all_data, "df": df}, f)
    print(f"Total time: {time.time() - t_0:.2f} s")
