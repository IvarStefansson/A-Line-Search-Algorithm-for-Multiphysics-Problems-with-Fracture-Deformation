import copy
import logging
import os
import time
from collections import namedtuple

import numpy as np
import porepy as pp

logger = logging.getLogger(__name__)


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
