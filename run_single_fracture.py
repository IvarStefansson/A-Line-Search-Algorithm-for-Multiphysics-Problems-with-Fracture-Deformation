import os

# Set number of threads to 1
os.environ["NUMBA_NUM_THREADS"] = "1"
import logging
import pickle
import time

import pandas as pd
import porepy as pp
from porepy.applications.md_grids.model_geometries import CubeDomainOrthogonalFractures

import model_setup
from common_params import algorithm_names, algorithms, params
from contact_mechanics_utilities import run_and_report_single
from visualisation import heatmap, strip_leading_zero_scientific

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    characteristic_displacement_scalings = [1e-4, 1e-2, 1, 1e2, 1e4]
    t_0 = time.time()
    dilation_angles = [0.1, 0.2]
    for nc in [6, 12]:
        for phys in range(2):
            phys_name, phys_model = model_setup.physical_models()[phys]
            base_name = "single_fracture/" + phys_name + f"_nc_{nc}/"
            columns = []
            all_iterations = {}
            all_data = {}
            for dil in dilation_angles:
                params["material_constants"]["solid"]._constants["dilation_angle"] = dil

                for u_c in characteristic_displacement_scalings:
                    case_name = f"u_c_{u_c}_dil_{dil}"
                    u_sc = strip_leading_zero_scientific(u_c)
                    col_name = r"$u_c$" + f"={u_sc / 1e2}\n" + r"$\phi$" + f"={dil}"
                    columns.append(col_name)

                    logger.info(
                        f"\n\nRunning {phys_name} with nc={nc}, u={u_c} and dil={dil}"
                    )

                    params.update(
                        {
                            "time_manager": pp.TimeManager([0, 1e6], 1e6, True),
                            "characteristic_displacement_scaling": u_c,
                            "folder_name": "results/" + base_name + case_name,
                            "solver_statistics_file_name": "solver_statistics.json",
                            "q_in": -5e-5,
                            "nl_convergence_tol_res": 1e-6,
                            "nl_convergence_tol": 1e-10,
                            "meshing_arguments": {"cell_size": 1 / nc},
                        },
                    )

                    class Model(
                        CubeDomainOrthogonalFractures,
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

                    # Loop over combinations of local and global line search values.
                    for params_loc, name in zip(algorithms, algorithm_names):
                        data_loc, model = run_and_report_single(
                            Model,
                            params,
                            params_loc,
                        )
                        iters.append(data_loc["iterations"])
                        keys.append(name)

                        all_data[case_name + name] = data_loc
                    all_iterations[col_name] = iters

            save = True
            if save:
                df = pd.DataFrame(all_iterations, index=keys, columns=columns)
                heatmap(
                    df,
                    0,
                    params["max_iterations"],
                    file_name=base_name + "/iteration_table.png",
                    title=phys_name + f", h=1/{nc}",
                )
                # Split the table in two for the two dilation angles
                df1 = df.iloc[:, :5]
                df2 = df.iloc[:, 5:]
                for dfi, phi in zip([df1, df2], dilation_angles):
                    # Strip unwanted characters from column names
                    dfi.columns = [col[:7] + col[9:] for col in dfi.columns]
                    dfi.columns = [col[6:10] for col in dfi.columns]
                    title = phys_name + f", h=1/{nc}, $\\phi={phi}$"
                    save_file = base_name + "/iteration_table" + f"_phi_{phi}" + ".png"
                    heatmap(
                        dfi,
                        0,
                        params["max_iterations"],
                        save_file,
                        title,
                        figsize=(5.0, 3.5),
                        x_label="$u_c$",
                    )

                # Save data to file.
                with open(base_name + "/all_data.pkl", "wb") as f:
                    pickle.dump({"all_data": all_data, "df": df}, f)
            else:
                print(df)
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
