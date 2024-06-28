import copy
import logging
import os
import time
from collections import namedtuple

import numpy as np
import porepy as pp

from trust_region import NonlinearSolver

logger = logging.getLogger(__name__)
nonlinear_solvers = [
    "undamped",
    "global",
    "harmonic_mean",
    "local",
    "median",
]

globalization_methods = [
    "undamped",
    "min",
    "harmonic_mean",
    "max",
    "median",
]


def run_multiple_simulations(
    model_type, params, solvers, relaxation_factors, severities, severity_name
):
    """Run multiple simulations with different parameters.

    Parameters:
        model_type: Type of model to run.
        params: Parameters for the model.
        solvers: List of solvers to use.
        relaxation_factors: List of relaxation factors to use.
        severities: List of severities to use. We sum along this axis.
        severity_name: Name of the severity parameter.

    Returns:
        Dictionary of collected data.

    """
    summed_data = {}
    counter = 0
    for solver_name in solvers:
        if solver_name == "undamped":
            params["nonlinear_solver"] = pp.NewtonSolver
        else:
            params["nonlinear_solver"] = NonlinearSolver

        params["globalization_type"] = solver_name
        for relax_factor in relaxation_factors:
            params["constraint_weight_relaxation"] = relax_factor
            relax_iters = np.array([])
            relax_times = np.array([])
            counter += 1
            for severity in severities:
                params[severity_name] = severity

                params_loc = copy.deepcopy(params)
                model = model_type(params_loc)
                logger.info(
                    f"Running with solver {solver_name}, relax factor {relax_factor} and"
                    + f"severity {severity}"
                )
                t_0 = time.time()
                pp.run_time_dependent_model(model, params_loc)
                num_iters = model.collected_data["num_nonlinear_iterations"][1:]
                if np.any(num_iters > params["max_iterations"]):
                    logger.warning(
                        f"Did not converge in {params['max_iterations']} iterations."
                    )
                    relax_iters = np.append(relax_iters, -1)
                else:
                    relax_iters = np.append(relax_iters, np.sum(num_iters))

                relax_times = np.append(relax_times, time.time() - t_0)

            summed_data[str(counter)] = {
                "iterations": relax_iters,
                "relax_factor": relax_factor,
                "solver": solver_name,
                "severities": np.array(severities),
                "solving_times": relax_times,
            }
            if solver_name == "undamped":
                # No need to run for different relax factors as these are not
                # used in the undamped case.
                break
    return summed_data


def run_and_report_single(model_type, params, update_params: dict[str, pp.number]):
    """Run a single simulation and report the results.

    Parameters:
        model_type: Type of model to run.
        params: Parameters for the model.
        update_params: List of parameters to update.

    Returns:
        Dictionary of collected data.

    """
    data = {}
    params_loc = copy.deepcopy(params)
    params_loc.update(update_params)
    model = model_type(params_loc)

    spec_msg = "Running with specs: \n"
    for name, val in update_params.items():
        spec_msg += f"{name}: {val} \n"
    logger.info(spec_msg)
    t_0 = time.time()
    # Purge previously stored vtu files
    pth = params_loc["folder_name"]
    if os.path.exists(pth):
        for f in os.listdir(pth):
            if f.endswith(".vtu") or f.endswith(".pvd"):
                os.remove(os.path.join(pth, f))
    debug = False
    if debug:
        pp.run_time_dependent_model(model, params_loc)
        num_iters = model.nonlinear_solver_statistics.num_iteration
    else:
        try:
            pp.run_time_dependent_model(model, params_loc)
            num_iters = model.nonlinear_solver_statistics.num_iteration
            error = model.nonlinear_solver_statistics.residual_norms[-1]
            if np.isnan(error) or np.isinf(error):
                num_iters = -200
                data["status"] = "Div"
        except ValueError as e:
            logger.warning(f"Value error: {e}")
            num_iters = -200
            data["status"] = "Div"
        except RuntimeError as e:
            logger.warning(f"Runtime error: {e}")
            num_iters = -200
            data["status"] = "Div"
    if num_iters == params_loc["max_iterations"] + 1:
        logger.warning(
            f"Did not converge in {params_loc['max_iterations']} iterations."
        )
        data["status"] = "MaxIt"
        # data["iterations"] = -num_iters
    if "status" not in data:
        data["status"] = "Conv"
    data["iterations"] = num_iters
    data["errors"] = model.nonlinear_solver_statistics.residual_norms
    data["solving_time"] = time.time() - t_0
    return data, model


def run_multiple_simulations_multifrac(model_type, params, axes: namedtuple):
    """Run multiple simulations with different parameters.

    Parameters:
        model_type: Type of model to run.
        params: Parameters for the model.
        axes: List of axes to visualize along. The last axis is plotted along the x axis.
            First axis is represented by line colors, (optional) second axis is
            represented by line markers. The plot's y axis corresponds to number
            of iterations.

    Returns:
        List of dictionaries of collected data.

    """

    def recursive_loop_level(
        params, axes, level, name_vals, report_data, iters=None, times=None
    ):
        if level == len(axes):
            # We have reached the last level. Run the model and collect data.
            params_loc = copy.deepcopy(params)
            model = model_type(params_loc)
            info = "Running with specs: \n"
            for name, val in name_vals.items():
                info += f"{name}: {val}, \n"
            logger.info(info)
            t_0 = time.time()
            debug = False
            if debug:
                pp.run_time_dependent_model(model, params_loc)
            else:
                try:
                    pp.run_time_dependent_model(model, params_loc)
                except ValueError as e:
                    logger.warning(f"Value error: {e}")
                    iters = np.append(iters, -10)
                    times = np.append(times, time.time() - t_0)
                    return iters, times
            num_iters = model.collected_data["num_nonlinear_iterations"][1:]
            if np.any(num_iters > params["max_iterations"]):
                logger.warning(
                    f"Did not converge in {params['max_iterations']} iterations."
                )
                iters = np.append(iters, -1)
            else:
                iters = np.append(iters, np.sum(num_iters))

            times = np.append(times, time.time() - t_0)

            return iters, times
        iters = np.array([])  # Overwritten until last level
        times = np.array([])
        for val in axes[level]:
            # if level == len(axes) - 1:

            name = axes._fields[level]
            name_vals.update({name: val})

            if any("undamped" == v for v in name_vals.values()):
                # Check if we have already covered the undamped for this set of parameters.
                # Disregard the damping parameters.
                damping_param_names = [
                    "relax_factor",
                    "interaction_region_depth",
                    "initial_globalization_type",
                ]
                covered = False
                for covered_name_vals in report_data:
                    if covered_name_vals["initial_globalization_type"] != "undamped":
                        continue
                    for this_name, this_val in name_vals.items():
                        if this_name in damping_param_names:
                            continue
                        if covered_name_vals[this_name] != this_val:
                            break
                    else:
                        covered = True
                        break

                if covered:
                    continue

            if name == "cell_size":
                params["meshing_arguments"]["cell_size"] = val
                params["meshing_arguments"]["cell_size_fracture"] = val / 2
            else:
                params[name] = val
            iters, times = recursive_loop_level(
                params, axes, level + 1, name_vals, report_data, iters, times
            )

            damping_param_names = [
                "relax_factor",
                "interaction_region_depth",
                "initial_globalization_type",
            ]

        if level == len(axes):
            data_loc = {
                "iterations": iters,
                "solving_times": times,
            }
            data_loc.update(name_vals)
            report_data.append(data_loc)
        if level == 0:
            return report_data
        else:
            return iters, times

    report_data = recursive_loop_level(params, axes, 0, {}, [])
    return report_data
