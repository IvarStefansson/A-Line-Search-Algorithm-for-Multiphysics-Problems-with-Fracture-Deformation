"""Module for line search algorithms for nonlinear solvers.

Main active classes, combined in the NonlinearSolver class:
- LineSearchNewtonSolver - extends NewtonSolver with line search and implements a basic
    line search based on the residual norm.
- SplineInterpolationLineSearch - implements a line search based on spline interpolation.
- ConstraintLineSearch - implements a line search based on constraint functions for contact
    mechanics.

The functionality is invoked by specifying the solver in the model parameters, e.g.:

    ```python
    model.params["nonlinear_solver"] = NonlinearSolver
    ```

The solver can be further customized by specifying parameters in the model parameters. It also
requires implementation of the constraint functions in the model as methods called
"opening_indicator" and "sliding_indicator", see model_setup.ContactIndicators.

"""

import logging
from typing import Callable, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps
import scipy.stats
from scipy import optimize

logger = logging.getLogger(__name__)


class LineSearchNewtonSolver(pp.NewtonSolver):
    """Class for relaxing a nonlinear iteration update using a line search.

    This class extends the iteration method to include a line search and implements
    a line search based on the residual norm.

    """

    def iteration(self, model) -> np.ndarray:
        """A single nonlinear iteration.

        Add line search to the iteration method.
        """
        dx = super().iteration(model)
        relaxation_vector = self.nonlinear_line_search(model, dx)

        # Update the solution
        sol = relaxation_vector * dx
        model._current_update = sol
        return sol

    def nonlinear_line_search(self, model, dx: np.ndarray) -> np.ndarray:
        """Perform a line search along the Newton step.

        Parameters:
            model: The model.
            dx: The Newton step.

        Returns:
            The step length vector, one for each degree of freedom.

        """
        return self.residual_line_search(model, dx)

    def residual_line_search(self, model, dx: np.ndarray) -> np.ndarray:
        """Compute the relaxation factors for the current iteration.

        The relaxation factors are computed based on the current Newton step.
        The relaxation factors are computed for each degree of freedom.
        """
        if not model.params.get("Global_line_search", False):
            return np.ones_like(dx)

        def objective_function(weight):
            return self.objective_function(model, dx, weight)

        tol = 1e-1
        f_0 = objective_function(0)
        f_1 = objective_function(1)
        if f_1 < model.params["nl_convergence_tol_res"] or (f_1 < f_0 / 1e4):
            # The objective function is zero at the Newton step. This means that the
            # Newton step is a minimum of the objective function. We can use the
            # Newton step without any relaxation.
            return np.ones_like(dx)

        def f_terminate(vals):
            return vals[-1] > vals[-2]

        alpha = self.recursive_weight_from_sampling(
            0, 1, f_terminate, objective_function, f_0, f_1, 3, tol
        )
        # safeguard against zero weights
        return np.maximum(alpha, tol / 10) * np.ones_like(dx)

    def recursive_weight_from_sampling(
        self,
        a: float,
        b: float,
        condition_function: Callable[[Callable], bool],
        function: Callable,
        f_a: Optional[float] = None,
        f_b: Optional[float] = None,
        num_steps: int = 5,
        step_size_tolerance: float = 1e-1,
    ) -> float:
        """Recursive function for finding a weight satisfying a condition.

        The function is based on sampling the function in the interval [a, b] and
        recursively narrowing down the interval until the interval is smaller than the
        tolerance and the condition is satisfied. It returns the smallest tested value
        not satisfying the condition.

        Parameters:
            a, b: The interval.
            condition_function: The condition function. It takes a function as argument and
                returns True if the condition is satisfied, indicating that the recursion
                should be terminated.
            function: The function to be tested. Returns a scalar or vector, must be compatible
                with the condition function's parameter.
            f_a: The value of the function at a. If not given, it is computed.
            num_steps: The number of sampling points in the interval [a, b].
            step_size_tolerance: The tolerance for the step size. If the step size is smaller
                than this, the recursion is terminated.

        Returns:
            The smallest tested value not satisfying the condition.

        """
        if f_a is None:
            f_a = function(a)
        terminate_condition = False
        sampling_points = np.linspace(a, b, num_steps)
        step_size = (b - a) / (num_steps - 1)
        f_vals = [f_a]
        for c in sampling_points[1:]:
            if np.isclose(c, b) and f_b is not None:
                f_c = f_b
            else:
                f_c = function(c)
            f_vals.append(f_c)
            terminate_condition = condition_function(f_vals)
            if not terminate_condition:
                f_a = f_c
                a = c
            else:
                # There is a local minimum in the narrowed-down interval [a, c]
                if step_size > step_size_tolerance:
                    # Find it to better precision
                    return self.recursive_weight_from_sampling(
                        a,
                        c,
                        condition_function,
                        function,
                        f_a=f_a,
                        num_steps=num_steps,
                        step_size_tolerance=step_size_tolerance,
                    )
                else:
                    # We're happy with the precision, return the minimum
                    return c

        # We went through the whole interval without finding a local minimum.
        # Thus, we assume that the sampled point in [c, b]. If we have reached
        # the tolerance, we return b. Otherwise, we search in [c, b].
        if step_size < step_size_tolerance:
            return b
        else:
            return self.recursive_weight_from_sampling(
                sampling_points[-2],
                b,
                condition_function,
                function,
                f_a=f_vals[-2],
                num_steps=num_steps,
                step_size_tolerance=step_size_tolerance,
            )

    def objective_function(self, model, dx: np.ndarray, weight: float) -> float:
        """Compute the objective function for the current iteration.

        The objective function is the norm of the residual.
        """
        # Define objective function and do simple line search along the Newton step.
        # The objective function is the norm of the residual.
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        residual = model.equation_system.assemble(
            state=x_0 + weight * dx, evaluate_jacobian=False
        )
        return np.linalg.norm(residual)


class SplineInterpolationLineSearch:
    """Class for computing the relaxation factors based on spline interpolation.

    This class could be seen as a tool for the technical step of actually
    performing a line search. It also specifies that this choice should be used
    for the objective function/residual and constraint weights.


    """

    def objective_function_weights(self, model, dx: np.ndarray) -> np.ndarray:
        """Specify that the objective function is computed based on spline interpolation."""
        x_0 = model.equation_system.get_variable_values(iterate_index=0)

        def f(w):
            r = model.equation_system.assemble(
                state=x_0 + w * dx, evaluate_jacobian=False
            )
            return np.dot(r.T, r)

        w, _, _ = self.recursive_spline_interpolation(
            0, 1, f, num_pts=5, interval_target_size=1e-3
        )
        return w

    def compute_constraint_violation_weights(
        self,
        model,
        solution_update: np.ndarray,
        region_function: pp.ad.Operator,
        crossing_inds: np.ndarray,
        f_0: np.ndarray,
        max_weight: float = 1,
        interval_target_size=1e-3,
    ) -> np.ndarray:
        """Specify that the constraint weights are computed based on spline interpolation."""
        if not np.any(crossing_inds):
            return 1
        # If the indicator has changed, we need to compute the relaxation factors. We do this by
        # recursively narrowing down the interval until the interval is smaller than the tolerance
        # using a spline interpolation.
        a, b = 0, max_weight
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        f_0 = f_0[crossing_inds]
        f_1 = region_function.value(model.equation_system, x_0 + solution_update * b)[
            crossing_inds
        ]

        def f(x):
            return region_function.value(
                model.equation_system, x_0 + solution_update * x
            )[crossing_inds]

        # Compute zero crossing and the minimum interval in which it is assumed to lie
        return_all = not self.use_fracture_minimum

        alpha, a, b = self.recursive_spline_interpolation(
            a,
            b,
            f,
            f_0,
            f_1,
            interval_target_size=interval_target_size,
            method="roots",
            return_all=return_all,
        )
        return alpha

    def recursive_spline_interpolation(
        self,
        a: float,
        b: float,
        function: Callable,
        f_a: Optional[float] = None,
        f_b: Optional[float] = None,
        num_pts: int = 5,
        interval_target_size: float = 1e-1,
        method="minimize_scalar",
        return_all=False,
    ) -> tuple[float, float, float]:
        """Recursive function for finding a weight satisfying a condition.

        Returns both the optimum/root (see method) and the minimal interval in
        which it is assumed to lie.

        Parameters:
            a, b: The interval.
            function: The function to be tested. Returns a scalar or vector defined on
                the interval [a, b].
            f_a: The value of the function at a. If not given, it is computed.
            f_b: The value of the function at b. If not given, it is computed.
            num_pts: The number of sampling points in the interval [a, b].
            step_size_tolerance: The tolerance for the step size. If the step size is smaller
                than this, the recursion is terminated.
            method: The method for finding the minimum of the spline. Either "minimize_scalar"
                or "roots".
            return_all: If True, return roots/minima for all vector outputs. Else, returns minimum
                weight. Experimental.

        Returns:
            Tuple containing:
                The minimum of the function in the interval [a, b].
                The lower bound of the interval in which the minimum is assumed to lie.
                The upper bound of the interval in which the minimum is assumed to lie.
        """
        counter = 0
        while b - a > interval_target_size or counter < 1:
            if return_all:
                num_pts *= 3
            alpha, x, y = self.optimum_from_spline(
                function,
                a,
                b,
                f_a,
                f_b,
                num_pts=num_pts,
                method=method,
                return_all=return_all,
            )
            if return_all:
                return alpha, x, y
            x = np.linspace(a, b, num_pts)
            # Find the indices on either side of alpha
            ind = np.searchsorted(x, alpha)
            if ind == 0:
                b = x[1]
                f_b = y[1]
            elif ind == num_pts:
                a = x[ind - 1]
                f_a = y[ind - 1]
            else:
                a = x[ind - 1]
                b = x[ind]
                f_a = y[ind - 1]
                f_b = y[ind]
            counter += 1

        return alpha, a, b

    def optimum_from_spline(
        self,
        f: Callable,
        a: float,
        b: float,
        f_a=None,
        f_b=None,
        num_pts: int = 5,
        method="minimize_scalar",
        return_all=False,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the minimum/root of the spline interpolation of the function.

        Parameters:
            f: The function to be interpolated.
            a, b: The interval.
            f_a: The value of the function at a. If not given, it is computed.
            f_b: The value of the function at b. If not given, it is computed.
            num_pts: The number of sampling points in the interval [a, b].
            method: The method for finding the minimum of the spline. Either "minimize_scalar"
                or "roots".
            return_all: If True, return roots/minima for all vector outputs. Else, returns minimum
                weight. Experimental.

        Returns:
            Tuple containing:
                The minimum of the function in the interval [a, b].
                The x values of the spline interpolation.
                The y values of the spline interpolation.

        """

        x = np.linspace(a, b, num_pts)
        y_list = []

        for pt in x:
            if f_a is not None and np.isclose(pt, a):
                f_pt = f_a
            elif f_b is not None and np.isclose(pt, b):
                f_pt = f_b
            else:
                f_pt = f(pt)
            if np.any(np.isnan(f_pt)):
                # If we get overflow, truncate the x vector
                x = x[: np.where(x == pt)[0][0]]
                break
            # Collect function values, scalar or vector
            y_list.append(f_pt)
        if isinstance(y_list[0], np.ndarray):
            y = np.vstack(y_list)
        else:
            y = np.array(y_list)

        def compute_and_postprocess_single(poly, a: float, b: float) -> float:
            if method == "minimize_scalar":
                minimum = scipy.optimize.minimize_scalar(
                    lambda s: poly(s), bounds=[a, b], method="bounded"
                )
                min_x = minimum.x
            elif method == "roots":
                min_x = poly.roots()
            if min_x.size == 0:
                return b
            else:
                # Find smallest root inside [a, b]
                min_x = min_x[(min_x >= a) & (min_x <= b)]
                if min_x.size == 0:
                    return b
                else:
                    return np.min(min_x)

        # Find minima of the spline

        if isinstance(y_list[0], np.ndarray):
            all_minima = []
            for i in range(y.shape[1]):
                poly = scipy.interpolate.PchipInterpolator(x, y[:, i])
                this_min = compute_and_postprocess_single(poly, a, b)
                all_minima.append(this_min)
            if return_all:
                return all_minima, x, y
            alpha = np.min(all_minima)
        else:
            poly = scipy.interpolate.PchipInterpolator(x, y)
            alpha = compute_and_postprocess_single(poly, a, b)

        return alpha, x, y


class ConstraintLineSearch:
    """Class for computing relaxation weights based on constraint functions
    for contact mechanics.

    The contract with the Model class is that the constraint functions are
    defined in the model as Operator returning methods called "opening_indicator"
    and "sliding_indicator".

    """

    @property
    def use_fracture_minimum(self):
        return True

    @property
    def min_weight(self):
        """Minimum weight for the relaxation weights."""
        return 1e-12

    def nonlinear_line_search(self, model, dx: np.ndarray) -> np.ndarray:
        """Perform a line search along the Newton step.

        First, call super method using the global residual as the objective function.
        Then, compute the constraint weights.

        Parameters:
            model: The model.
            dx: The Newton step.

        Returns:
            The step length vector, one for each degree of freedom.

        """
        residual_weight = self.residual_line_search(model, dx)
        if model.params.get("Local_line_search", False):
            return self.constraint_line_search(model, dx, residual_weight.min())
        else:
            return residual_weight

    def constraint_line_search(
        self, model, dx: np.ndarray, max_weight: float = 1
    ) -> np.ndarray:
        """Perform line search along the Newton step for the constraints.

        This method defines the constraint weights for the contact mechanics and
        how they are combined to a global weight. For more advanced combinations,
        this method can be overridden or the stored weights can be accessed elsewhere.

        Parameters:
            model: The model.
            dx: The Newton step.
            max_weight: The maximum weight for the constraint weights.

        Returns:
            The step length vector, one for each degree of freedom.

        """

        subdomains = model.mdg.subdomains(dim=model.nd - 1)

        sd_weights = {}
        global_weight = max_weight
        for sd in subdomains:
            sd_list = [sd]
            # Compute the relaxation factors for the normal and tangential
            # component of the contact mechanics.
            normal_weights = self.constraint_weights(
                model,
                dx,
                model.opening_indicator(sd_list),
                max_weight=max_weight,
            )
            tangential_weights = self.constraint_weights(
                model,
                dx,
                model.sliding_indicator(sd_list),
                max_weight=np.minimum(max_weight, normal_weights).min(),
            )
            # For each cell, take minimum of tangential and normal weights
            combined_weights = np.vstack((tangential_weights, normal_weights))
            min_weights = np.min(combined_weights, axis=0)
            # Store the weights for the subdomain. This facilitates more advanced
            # combinations of the constraint weights, e.g. using local weights for
            # each cell.
            sd_weights[sd] = min_weights
            model.mdg.subdomain_data(sd).update({"constraint_weights": min_weights})
            # Combine the weights for all subdomains to a global minimum weight.
            global_weight = np.minimum(global_weight, min_weights.min())
        # Return minimum of all weights.
        weight = np.ones_like(dx) * global_weight
        return weight

    def constraint_weights(
        self,
        model,
        solution_update: np.ndarray,
        constraint_function: pp.ad.Operator,
        max_weight: float = 1,
    ) -> np.ndarray:
        """Compute weights for a given constraint.

        This method specifies the algorithm for computing the constraint weights:
        - Identify the indices where the constraint function has changed sign.
        - Compute the relaxation factors for these indices, allowing transition beyond
            zero by a tolerance given by the constraint_violation_tolerance parameter.
        - Reassess the constraint function at the new weights and repeat the process if
            too many indices are still transitioning.

        Parameters:
            model: The model.
            solution_update: The Newton step.
            constraint_function: The constraint function.
            max_weight: The maximum weight for the constraint weights.

        Returns:
            The step length vector, one for each degree of freedom of the constraint.

        """
        # If the sign of the function defining the regions has not changed, we
        # use unitary relaxation factors
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        violation_tol = model.params.get("constraint_violation_tolerance", 3e-1)
        relative_cell_tol = model.params.get(
            "relative_constraint_transition_tolerance", 2e-1
        )
        f_1 = constraint_function.value(
            model.equation_system, x_0 + max_weight * solution_update
        )
        weights = max_weight * np.ones(f_1.shape)
        f_0 = constraint_function.value(model.equation_system, x_0)
        active_inds = np.ones(f_0.shape, dtype=bool)
        for i in range(10):
            # Only consider dofs where the constraint violation has changed sign
            violation = violation_tol * np.sign(f_1)
            f = constraint_function - pp.wrap_as_dense_ad_array(violation)
            # Absolute tolerance should be safe, as constraints are assumed to be
            # scaled to approximately 1.
            roundoff = 1e-8
            inds = np.logical_and(np.abs(f_1) > violation_tol, f_0 * f_1 < -roundoff)
            if i > 0:  # Ensure at least one iteration.
                if sum(active_inds) < max(1, relative_cell_tol * active_inds.size):
                    # Only a few indices are active, and the set of active indices
                    # does not contain any new indices. We can stop the iteration.
                    break

                else:
                    logger.info(
                        f"Relaxation factor {weight} is too large for {sum(active_inds)}"
                        + " indices. Reducing constraint violation tolerance."
                    )

            f_0_v = f_0 - violation
            crossing_weight = self.compute_constraint_violation_weights(
                model,
                solution_update,
                f,
                inds,
                f_0_v,
                max_weight=max_weight,
                interval_target_size=1e-3,
            )
            weight = np.clip(crossing_weight, a_max=max_weight, a_min=self.min_weight)
            weights[inds] = weight

            if not self.use_fracture_minimum:
                break  # Experimental.
            # Check how many indices are active for current weight
            f_1 = constraint_function.value(
                model.equation_system, x_0 + weight * solution_update
            )
            active_inds = np.logical_and(
                np.abs(f_1) > violation_tol, f_0 * f_1 < -roundoff
            )
            if i == 9:
                logger.info(
                    "Maximum number of iterations reached. "
                    + "Returning current weights."
                )
            max_weight = weight
            violation_tol = violation_tol / 2

        return weights


class NonlinearSolver(
    ConstraintLineSearch,
    SplineInterpolationLineSearch,
    LineSearchNewtonSolver,
):
    pass


# More advanced/experimental line search algorithms, including functionality for
# local weight assignment. These are not used in the current implementation.
class LineSearchStrongWolfe:
    def objective_function_weights(self, model, dx: np.ndarray) -> np.ndarray:

        return self.line_search(model, dx)

    def line_search(self, model, dx: np.ndarray) -> np.ndarray:
        """Perform a line search along the Newton step.

        The line search algorithm is based on the strong Wolfe conditions.

        Parameters:
            model: The model.
            dx: The Newton step.

        Returns:
            The step length.

        """
        # Define objective function and do simple line search along the Newton step.
        # The objective function is the norm of the residual.
        x_0 = model.equation_system.get_variable_values(iterate_index=0)

        # Define your residual function r(x)
        def merit_function(x):
            r = model.equation_system.assemble(state=x, evaluate_jacobian=False)
            return np.dot(r.T, r)

        f_0 = merit_function(x_0)
        # f_eps = merit_function(x_0 + self.min_weight * dx)
        # if f_eps > f_0:
        #     warnings.warn(
        #         "dx is not a descent direction. Returning alpha = minimum weight."
        #     )
        #     return self.min_weight
        f_1 = merit_function(x_0 + dx)
        if f_1 < model.params["nl_convergence_tol_res"] or (f_1 < f_0 / 1e4):
            # The objective function is zero at the Newton step. This means that the
            # Newton step is a minimum of the objective function. We can use the
            # Newton step without any relaxation.
            return 1

        # Define the Jacobian of the merit function
        def merit_function_jacobian(x):
            j, r = model.equation_system.assemble(state=x)
            return 2 * j.T @ r

        def merit_and_jacobian(x):
            r, j = model.equation_system.assemble(state=x)
            return np.dot(r.T, r), 2 * j.T @ r

        alpha = np.array([1e-2])

        # Define bounds bounds = Bounds([0], [1]) res = minimize(merit_function,
        # x0, method='L-BFGS-B', jac=merit_function_jacobian, bounds=bounds)

        # print("Optimal parameters:", res.x)
        # c values from Nocedal and Wright, Numerical Optimization
        f = merit_function
        df = merit_function_jacobian
        alpha, *_ = optimize.line_search(
            f, df, x_0, dx, c1=1e-4, c2=0.9, maxiter=4, old_fval=f_0
        )
        alpha = None

        if alpha is None or alpha <= self.min_weight:

            a = 0

            a_p, b, num_pts = max(a, self.min_weight), 1, 10
            # Create list of logarithmically spaced points in the interval [a,
            # b]
            x1 = self.min_weight
            x_list = [a]
            coeff = np.exp(-np.log(x1) / num_pts)
            while x1 < b:
                x_list.append(x1)
                x1 = x1 * coeff
            x = np.array(x_list)
            x = x / x[-1]
            x_log = np.logspace(np.log10(a_p), np.log10(b), num_pts)[:-1]
            x_lin = np.linspace(x_log[-1], b, num_pts)
            x = np.concatenate((x_log, x_lin[1:]))
            y_list = []

            def f(w):
                r = model.equation_system.assemble(
                    state=x_0 + w * dx, evaluate_jacobian=False
                )
                return np.dot(r.T, r)

            for pt in x:
                f_pt = f(pt)
                # Collect function values, scalar or vector
                y_list.append(f_pt)
            y = np.array(y_list)

            poly = scipy.interpolate.PchipInterpolator(x, y)

            def fp(x):
                return poly(x)

            dpx = poly.derivative()

            def dfp(x):
                return dpx(x)

            alpha_p, *_ = optimize.line_search(
                fp,
                dfp,
                np.array([0]),
                np.array([1]),
                c1=1e-4,
                c2=0.9,
                maxiter=50,
            )
            alpha = alpha_p

        if alpha is None:
            alpha = self.min_weight
        elif isinstance(alpha, np.ndarray):
            print(alpha)
            raise ValueError("Alpha is an array")
            # alpha = alpha[0]
        alpha = np.clip(alpha, 0, 1)
        return alpha


class LocalConstraintLineSearch(ConstraintLineSearch):
    @property
    def use_fracture_minimum(self):
        return self.params.get("use_fracture_minimum", True)

    # def constraint_weights(self, model, dx: np.ndarray, max_weight: float = 1) -> np.ndarray:
    #     super().constraint_weights(model, dx, max_weight)
    #     self._identify_connected_sets(model, subdomains)

    #     for k in sd_weights.keys():
    #         sd_weights[k] = np.minimum(sd_weights[k], max_weight)
    #     # Having found the connected sets, we can compute the weights for each set
    #     for connected_set in self._connected_sets.keys():
    #         weight_set = 1

    #         for sd in connected_set:
    #             weight_set = np.minimum(weight_set, np.min(sd_weights[sd]))
    #         for sd in connected_set:
    #             sd_weights[sd][:] = weight_set
    #         self._connected_sets[connected_set]["weight"] = weight_set

    def _identify_connected_sets(self, model, subdomains: list[pp.Grid]) -> list[list]:
        """TODO: Use networkx to identify connected sets."""
        # Traverse subdomains to check if they intersect with another subdomain/fracture.
        connected_sets = {}

        # fix. also consider traversing intersections below.
        for sd in subdomains:
            sd_list = [sd]
            interfaces = model.subdomains_to_interfaces(sd_list, codims=[1])
            if len(interfaces) == 1:  # Only one interface (with the matrix)
                # sd is does not intersect with any other subdomain
                connected_sets[tuple([sd])] = {}
                continue
            # Check if sd is contained in a previously found set
            contained = False
            for connected_set in connected_sets.keys():
                if sd in connected_set:
                    contained = True
                    break
            if contained:
                continue
            # sd is not contained in a previously found set, so we need to find the
            # connected set of subdomains.
            # Interfaces between fracture and intersection
            intersection_intfs = [
                intf for intf in interfaces if intf.dim == model.nd - 2
            ]
            # All subdomains of the intersection interfaces
            sds_of_intersection_intfs = model.interfaces_to_subdomains(
                intersection_intfs
            )
            # Find only lower-dimensional subdomains, i.e., the intersection
            # subdomains:
            intersection_sds = [
                sd_ for sd_ in sds_of_intersection_intfs if sd_.dim < sd.dim
            ]
            # Find all interfaces of the intersections
            intfs_of_intersections = model.subdomains_to_interfaces(
                intersection_sds, codims=[1]
            )
            # Filter out the lower-dimensional interfaces (i.e., the
            # intersection points)
            fracture_intfs_of_intersections = [
                intf for intf in intfs_of_intersections if intf.dim == model.nd - 2
            ]
            fracture_intf_neighbours = model.interfaces_to_subdomains(
                fracture_intfs_of_intersections
            )
            # Traverse back up to fractures. This should be all fractures
            # neighbouring any intersection contained in sd.
            fractures_of_intersection = [
                sd_ for sd_ in fracture_intf_neighbours if sd_.dim == model.nd - 1
            ]
            # Add this set of connected subdomains to the list.
            connected_sets[tuple(fractures_of_intersection)] = {}
        self._connected_sets = connected_sets


class LinearSystemBasedDistribution:
    """Distribute the relaxation factors based on the linear system.

    The relaxation factors are distributed based on regions identified by
    repeated left multiplication by the linear system matrix applied to Dirac
    type vectors for the fracture regions.

    Algorithm:
        1. Identify the regions by repeated left multiplication by the linear
           system matrix applied to Dirac type vectors for the fracture regions.
        2. Compute the relaxation factors for each region.
        3. Distribute the relaxation factors to the DOFs in each region.


    """

    def _identify_interaction_regions(self, model):
        """Identify the regions by repeated left multiplication by the linear
        system matrix applied to Dirac type vectors for the fracture regions.

        For each region, construct a Dirac vector for all (traction) DOFs. Then
        multiply by the linear system matrix N times. The resulting vector will have
        non-zero entries for all DOFs that interact with the region.

        Parameters:
            model: The model.
            connected_sets: The connected sets of subdomains.

        """
        eq_sys: pp.EquationSystem = model.equation_system
        depth = self.params.get("interaction_region_depth", 2)
        if depth <= 0:
            return
        mat = model.linear_system[0]
        # Avoid overflow/numerical issues by using unitary data
        mat.data = np.ones_like(mat.data)
        # mat_to_power_n = mat

        # for _ in range(depth - 1):
        #     mat_to_power_n = mat_to_power_n @ mat
        for connected_sds, sd_dict in self._connected_sets.items():
            # Construct the Dirac vector for the fracture DOFs
            # if by_vars:
            variables = [
                v
                for v in eq_sys.variables
                if v.domain in connected_sds
                # and v.name == model.contact_traction_variable
            ]
            var_dofs = eq_sys.dofs_of(variables)

            dirac_vars = np.zeros(eq_sys.num_dofs())
            dirac_vars[var_dofs] = 1

            # Multiply by the linear system matrix
            # region_vector = mat_to_power_n * dirac > 0
            # Store the region vector
            for _ in range(depth):
                dirac_vars = mat * dirac_vars
            # else:
            # By equation indices
            contact_inds = np.hstack(
                (
                    model.equation_system.assembled_equation_indices[
                        "tangential_fracture_deformation_equation"
                    ],
                    model.equation_system.assembled_equation_indices[
                        "normal_fracture_deformation_equation"
                    ],
                )
            )

            nonzero_rows = contact_inds.astype(int)
            for _ in range(depth):
                zero_inds = np.setdiff1d(np.arange(eq_sys.num_dofs()), nonzero_rows)
                mat_loc = mat.copy()
                pp.matrix_operations.zero_rows(mat_loc, zero_inds)
                # Affected dofs are those columns that are not zeroed out
                nonzero_cols = mat_loc.nonzero()[1]
                unique_cols = np.unique(nonzero_cols)
                dirac = np.zeros(eq_sys.num_dofs())
                dirac[unique_cols] = 1
                nonzero_row_inds = mat @ dirac > 0
                nonzero_rows = nonzero_row_inds.nonzero()[0]
            region_vector = dirac

            sd_dict["region_vector"] = region_vector

            # Check for overlap with other regions
            for other_sd_dict in self._connected_sets.values():
                if "region_vector" not in other_sd_dict:
                    continue
                if np.any(
                    np.logical_and(region_vector, other_sd_dict["region_vector"])
                ):
                    # Assign minimum weight to both regions
                    min_weight = np.minimum(sd_dict["weight"], other_sd_dict["weight"])
                    sd_dict["weight"] = min_weight
                    other_sd_dict["weight"] = min_weight

    def relaxation_vector(self, model, residual_weight: float) -> np.ndarray:
        """Compute the relaxation vector for the current iteration."""
        domains = model.mdg.subdomains()

        def compute_cell_faces(intfs):
            for intf in intfs:
                intf.cell_faces = sps.block_diag(
                    [g.cell_faces for g in intf.side_grids.values()]
                )
                intf.num_faces = intf.cell_faces.shape[0]

        num_cells = np.cumsum([d.num_cells for d in domains])
        num_cells = np.insert(num_cells, 0, 0)

        interfaces = model.mdg.interfaces()
        num_cells_intfs = np.cumsum([i.num_cells for i in interfaces])
        num_cells_intfs = np.insert(num_cells_intfs, 0, 0)
        compute_cell_faces(interfaces)
        tr = pp.ad.Trace(domains, dim=1)
        tr_i = pp.ad.Trace(interfaces, dim=1)
        proj = pp.ad.MortarProjections(model.mdg, domains, interfaces, dim=1)
        # Subdomain cells to subdomain cells
        block_00 = (tr.inv_trace @ tr.trace).value(model.equation_system)
        # Interface cells to subdomain cells. Both primary and secondary, the
        # former via subdomain faces
        to_primary = (tr.inv_trace @ proj.mortar_to_primary_avg).value(
            model.equation_system
        )
        to_secondary = (proj.mortar_to_secondary_avg).value(model.equation_system)
        block_01 = to_primary + to_secondary
        block_10 = block_01.T
        # Interface cells to interface cells
        block_11 = (tr_i.inv_trace @ tr_i.trace).value(model.equation_system)
        # We're only interested in connections to other cells. Change entries to
        # ones
        for block in [block_00, block_01, block_10, block_11]:
            block.data = np.ones_like(block.data)
        full_mat = sps.bmat([[block_00, block_01], [block_10, block_11]]).tocsr()

        incr_factor = model.params.get("constraint_weight_relaxation", 1.0)
        depth = model.params.get("constraint_weight_depth", -1)

        all_connected_set_weights = (
            np.ones((full_mat.shape[0], len(self._connected_sets) + 1))
            * residual_weight
        )

        def weight_indices(g):
            if isinstance(g, pp.MortarGrid):
                ind = np.in1d(np.array(interfaces), g).nonzero()[0][0]
                indices = np.arange(num_cells_intfs[ind], num_cells_intfs[ind + 1])
                indices += num_cells[-1]
            else:
                ind = np.in1d(np.array(domains), g).nonzero()[0][0]
                indices = np.arange(num_cells[ind], num_cells[ind + 1])
            return indices

        i = 0
        for connected_sds, sd_dict in self._connected_sets.items():
            if depth == 0:
                continue
            _connected_weights = np.ones(full_mat.shape[0])

            # Find indices of the subdomains in full_mat
            _min_weight = sd_dict["weight"]
            for sd in connected_sds:
                weight_inds = weight_indices(sd)

                if self.use_fracture_minimum:
                    _connected_weights[weight_inds] = sd_dict["weight"]
                else:
                    _connected_weights[weight_inds] = model.mdg.subdomain_data(sd)[
                        "constraint_weights"
                    ]
                    _min_weight = np.minimum(
                        _min_weight,
                        model.mdg.subdomain_data(sd)["constraint_weights"].min(),
                    )
            if _min_weight == 1:
                continue
            # No need to go deeper than depth_loc s.t. incr_factor ** depth_loc * _min_weight = 1
            if incr_factor > 1:
                depth_loc = int(np.log(1 / _min_weight) / np.log(incr_factor))
                if depth > 0:
                    depth_loc = np.minimum(depth_loc, depth)
            else:
                if depth == -1:
                    # Indicates no limit on depth.
                    if incr_factor == 1:
                        # In this case, all weights are set to the minimum weight
                        all_connected_set_weights[:, i] = _min_weight
                        i += 1
                        continue
                    if incr_factor == 0:
                        # Interpret as not using constraint weights
                        continue

                depth_loc = depth
            connected_weights = np.ones((full_mat.shape[0], depth_loc + 1))
            connected_weights[:, 0] = _connected_weights
            affected_cells = np.zeros(full_mat.shape[0], dtype=bool)
            for d in range(1, depth_loc + 1):
                # inds = np.abs() > 0
                for c in range(full_mat.shape[0]):
                    # Find the nonzero entries in the row
                    # neighs = full_mat[c].nonzero()[1]
                    # Or, more efficient for csr matrix
                    neighs = full_mat.indices[
                        full_mat.indptr[c] : full_mat.indptr[c + 1]
                    ]

                    min_w = np.min(connected_weights[neighs, d - 1])
                    if np.any(connected_weights[neighs, d - 1] != 1):
                        affected_cells[c] = True
                    connected_weights[c, d] = np.clip(
                        min_w * incr_factor, a_max=1, a_min=0
                    )
                if np.all(connected_weights[:, d] >= 1):
                    break
            all_connected_set_weights[:, i] = np.min(connected_weights, axis=1)
            i += 1
        cellwise_weights = np.min(all_connected_set_weights, axis=1)
        for sd, data in model.mdg.subdomains(return_data=True):
            inds = weight_indices(sd)
            pp.set_solution_values(
                name="constraint_weights",
                values=cellwise_weights[inds],
                data=data,
                iterate_index=0,
            )
        # Distribute to all variables
        weights_all_vars = np.zeros(model.equation_system.num_dofs())
        for var in model.equation_system.variables:
            domain = var.domain
            weight_inds = weight_indices(domain)

            weights_local = cellwise_weights[weight_inds]
            dof_indices = model.equation_system.dofs_of([var])
            if var._cells > 1:
                weights_local = np.repeat(weights_local, var._cells)
            weights_all_vars[dof_indices] = weights_local
        # Report minimum weights
        logger.info(
            f"Residual weight {residual_weight:.2e}, "
            + f"actual min and max relaxation {weights_all_vars.min():.2e}, {weights_all_vars.max():.2e}."
        )
        return weights_all_vars

    def _relaxation_vector(self, model, residual_weight: float) -> np.ndarray:
        eq_sys = model.equation_system

        # The regions may overlap. We assign one relaxation factor to each DOF for each region.
        background_weights = np.ones(eq_sys.num_dofs()) * residual_weight
        logger_msg = f"Background weight: {residual_weight:.2e}.\n"
        relaxation_matrix = np.tile(
            np.atleast_2d(background_weights).T, len(self._connected_sets)
        )
        logger_msg += "Constraint weights: "
        for i, sd_dict in enumerate(self._connected_sets.values()):
            # Find the DOFs in the region
            region_dofs = sd_dict["region_vector"].nonzero()[0]

            # Assign the relaxation factor to the DOFs in the region
            relaxation_matrix[region_dofs, i] = sd_dict["weight"]
            logger_msg += f"{sd_dict['weight']:.2e}, "

        logger.info(logger_msg[:-2] + ".")
        # Take the minimum of the relaxation factors for each DOF
        relaxation_vector = np.min(relaxation_matrix, axis=1)
        return relaxation_vector
