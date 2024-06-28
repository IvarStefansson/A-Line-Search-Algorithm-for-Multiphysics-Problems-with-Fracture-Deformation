"""Visualize etc."""

import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import porepy as pp
import scipy.sparse as sps
from scipy.optimize import Bounds, minimize

logger = logging.getLogger(__name__)


class ExportScaledData:
    equation_system: pp.EquationSystem
    fluid: pp.FluidConstants

    def before_nonlinear_loop(self) -> None:
        self._did_converge = False
        self._errors = []
        super().before_nonlinear_loop()

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[float, bool, bool]:
        """Implements a convergence check, to be called by a non-linear solver.

        Parameters:
            solution: Newly obtained solution vector prev_solution: Solution obtained in
            the previous non-linear iteration. init_solution: Solution obtained from the
            previous time-step. nl_params: Dictionary of parameters used for the
            convergence check.
                Which items are required will depend on the convergence test to be
                # implemented.

        Returns:
            The method returns the following tuple:

            float:
                Error, computed to the norm in question.
            boolean:
                True if the solution is converged according to the test implemented by
                this method.
            boolean:
                True if the solution is diverged according to the test implemented by
                this method.

        """
        (
            residual_norm,
            nonlinear_increment_norm,
            converged,
            diverged,
        ) = super().check_convergence(
            nonlinear_increment, residual, reference_residual, nl_params
        )
        logger.info(
            f"Residual norm: {residual_norm:.2e} and increment norm: {nonlinear_increment_norm:.2e}."
        )
        self._errors.append(residual_norm)

        return residual_norm, nonlinear_increment_norm, converged, diverged

    def data_to_export(self):
        """Return data to be exported.

        Return type should comply with pp.exporter.DataInput.

        Returns:
            List containing all (grid, name, scaled_values) tuples.
        """
        data = super().data_to_export()
        eqs2 = (
            self.equation_system_2
            if hasattr(self, "equation_system_2")
            else self.equation_system
        )
        for dim in range(self.nd + 1):
            for sd in self.mdg.subdomains(dim=dim):
                if dim == self.nd - 1:
                    names = ["displacement_jump", "aperture"]
                    for n in names:
                        data.append((sd, n, self._evaluate_and_scale(sd, n, "m")))
                    data.append(
                        (
                            sd,
                            "contact_states",
                            self.report_on_contact_states([sd]),
                        )
                    )
                    val = self._evaluate_and_scale(
                        sd, "contact_traction", "Pa"
                    ) / self.characteristic_traction([sd]).value(self.equation_system)
                    data.append((sd, "fracture_traction", val))
                    val = self._evaluate_and_scale(
                        sd, "displacement_jump", "m"
                    ) / self.characteristic_displacement([sd]).value(
                        self.equation_system
                    )
                    data.append((sd, "scaled_displacement_jump", val))

                if hasattr(self, "residual_variable") and sd.dim == self.nd - 1:
                    data.append(
                        (
                            sd,
                            "residual_variable",
                            self.residual_variable([sd]).value(eqs2),
                        )
                    )
                data.append(
                    (
                        sd,
                        "constraint_weights",
                        pp.get_solution_values(
                            "constraint_weights",
                            self.mdg.subdomain_data(sd),
                            iterate_index=0,
                        ),
                    )
                )

        return data

    def report_on_contact_states(self, subdomains: list[pp.Grid] = None):
        """Report on the contact states of the fractures.

        Parameters:
            subdomains: List of subdomains to report on. If None, all fractures are
                considered.

        Returns:
            np.ndarray: Array of contact states, one for each fracture cell.

        """

        if subdomains is None:
            subdomains = self.mdg.subdomains(dim=self.nd - 1)

        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        contact_force_n = t_n.value(self.equation_system)
        opening = (u_n - self.fracture_gap(subdomains)).value(self.equation_system)
        c_num = self.contact_mechanics_numerical_constant(subdomains).value(
            self.equation_system
        )
        zerotol = 1e-12
        in_contact = (-contact_force_n - c_num * opening) > zerotol

        nd_vec_to_tangential = self.tangential_component(subdomains)

        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)

        # Combine the above into expressions that enter the equation
        ut_val = u_t.value(self.equation_system).reshape((self.nd - 1, -1), order="F")
        sliding = np.logical_and(np.linalg.norm(ut_val, axis=0) > zerotol, in_contact)
        # 0 sticking, 1 sliding, 2 opening
        return sliding + 2 * np.logical_not(in_contact)

    def after_nonlinear_failure(self) -> None:
        """Method to be called if the non-linear solver fails to converge.

        Parameters:
            solution: The new solution, as computed by the non-linear solver.
            errors: The error in the solution, as computed by the non-linear solver.
            iteration_counter: The number of iterations performed by the non-linear
                solver.

        """
        self.save_data_time_step()
        logger.info("Nonlinear iterations did not converge.")


class IterationExporting:
    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        # Setting export_constants_separately to False facilitates operations such as
        # filtering by dimension in ParaView and is done here for illustrative purposes.
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_iterations",
            folder_name=self.params["folder_name"],
            export_constants_separately=False,
        )

    def data_to_export_iteration(self):
        """Returns data for iteration exporting.

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        # The following is a slightly modified copy of the method
        # data_to_export() from DataSavingMixin.
        data = []
        variables = self.equation_system.variables
        eqs2 = (
            self.equation_system_2
            if hasattr(self, "equation_system_2")
            else self.equation_system
        )
        for var in variables:
            # Note that we use iterate_index=0 to get the current solution, whereas
            # the regular exporter uses time_step_index=0.
            scaled_values = self.equation_system.get_variable_values(
                variables=[var], iterate_index=0
            )
            units = var.tags["si_units"]
            values = self.fluid.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain, var.name, values))
        for name in ["displacement_jump", "aperture"]:
            for sd in self.mdg.subdomains(dim=self.nd - 1):
                vals = self._evaluate_and_scale(sd, name, "m")
                data.append((sd, name, vals))
        for sd, _data in self.mdg.subdomains(return_data=True):
            data.append(
                (
                    sd,
                    "constraint_weights",
                    pp.get_solution_values(
                        "constraint_weights",
                        _data,
                        iterate_index=0,
                    ),
                )
            )
        for sd in self.mdg.subdomains(dim=self.nd - 1):
            # Map interface displacement to fracture and take average of the two sides.
            intfs = self.mdg.subdomain_to_interfaces(sd)
            for intf in intfs:
                if intf.dim == self.nd - 1:
                    u_j = self._evaluate_and_scale(intf, "interface_displacement", "m")
                    proj = intf.mortar_to_secondary_avg(nd=self.nd)
                    u_f = proj @ u_j / 2
                    data.append((sd, self.displacement_variable, u_f))
                    break
            vals = self.report_on_contact_states([sd])
            data.append((sd, "contact_states", vals))
            for n in ["opening_indicator", "sliding_indicator"]:
                data.append((sd, n, self._evaluate_and_scale(sd, n, "1")))

            val = self._evaluate_and_scale(
                sd, "contact_traction", "Pa"
            ) / self.characteristic_traction([sd]).value(self.equation_system)
            data.append((sd, "fracture_traction", val))
            val = self._evaluate_and_scale(
                sd, "displacement_jump", "m"
            ) / self.characteristic_displacement([sd]).value(self.equation_system)
            data.append((sd, "scaled_displacement_jump", val))
            data.append(
                (
                    sd,
                    "friction_bound",
                    self._evaluate_and_scale(sd, "friction_bound", "Pa"),
                )
            )
            data.append(
                (
                    sd,
                    "t_c_estimate",
                    self._evaluate_and_scale(
                        sd, "characteristic_fracture_traction_estimate", "Pa"
                    ),
                )
            )

            if hasattr(self, "residual_variable"):
                data.append(
                    (
                        sd,
                        "residual_variable",
                        self.residual_variable([sd]).value(eqs2),
                    )
                )
        return data

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.

        Having a separate exporter for iterations avoids distinguishing between iterations
        and time steps in the regular exporter's history (used for export_pvd).

        """
        # To make sure the nonlinear iteration index does not interfere with the
        # time part, we multiply the latter by the next power of ten above the
        # maximum number of nonlinear iterations. Default value set to 10 in
        # accordance with the default value used in NewtonSolver
        n = self.params.get("max_iterations", 10)
        p = round(np.log10(n))
        r = 10**p
        if r <= n:
            r = 10 ** (p + 1)
        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(),
            time_dependent=True,
            time_step=self.nonlinear_solver_statistics.num_iteration
            + r * self.time_manager.time_index,
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution to
        iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()

    def prepare_simulation(self):
        """Prepare simulation."""
        super().prepare_simulation()
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()


class PostProcessing(
    ExportScaledData,
    pp.DiagnosticsMixin,
):
    """Combine classes used for post-processing."""
