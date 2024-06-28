import logging
from functools import partial
from typing import Any, Callable

import numpy as np
import porepy as pp
import scipy.sparse as sps

from boundary_conditions import (DisplacementBoundaryConditionsLinear,
                                 EnergyBoundaryConditionsDirEastWest,
                                 FluidFlowBoundaryConditionsDirEastWest)
from postprocessing import ExportScaledData, IterationExporting
from smoothing_model import (AdaptiveSmoothingExplicit,
                             AdaptiveSmoothingImplicit)

logger = logging.getLogger(__name__)


# Step and Heaviside functions
def heaviside(var, zerovalue: float = 0.0):
    zero_jac = 0
    if isinstance(var, pp.ad.AdArray):
        zero_jac = sps.csr_matrix(var.jac.shape)
    if isinstance(var, pp.ad.AdArray):
        return pp.ad.AdArray(np.heaviside(var.val, zerovalue), zero_jac)
    else:
        return np.heaviside(var, zerovalue)


def sign(var):
    zero_jac = 0
    if isinstance(var, pp.ad.AdArray):
        zero_jac = sps.csr_matrix(var.jac.shape)
    if isinstance(var, pp.ad.AdArray):
        return pp.ad.AdArray(np.sign(var.val), zero_jac)
    else:
        return np.sign(var)


class ScaledContactMechanics:

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Equation for the normal component of the fracture deformation.

        This constraint equation enforces non-penetration of opposing fracture
        interfaces.

        Parameters:
            subdomains: List of subdomains where the normal deformation equation is
            defined.

        Returns:
            Operator for the normal deformation equation.

        """

        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump
        t_n: pp.ad.Operator = (
            nd_vec_to_normal
            @ self.contact_traction(subdomains)
            / self.characteristic_traction(subdomains)
        )
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        # Maximum function
        num_cells: int = sum([sd.num_cells for sd in subdomains])

        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells), "zeros_frac")

        # The complimentarity condition
        c_num = self.contact_mechanics_numerical_constant(subdomains)
        # kkt
        f = pp.ad.Scalar(-1.0) * t_n  # >= 0
        g = c_num * (u_n - self.fracture_gap(subdomains))  # >= 0
        a = f - g
        b = zeros_frac

        equation: pp.ad.Operator = f - self._max(subdomains, a, b)

        if self.params.get("relative_deformation_equations", False):
            equation /= self.characteristic_fracture_traction_operator(subdomains)

        equation.set_name("normal_fracture_deformation_equation")
        return equation

    def _max(self, subdomains, a, b):
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        sharp_max = max_function(a, b)
        return sharp_max

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """
        Contact mechanics equation for the tangential constraints.

        The function reads
        .. math::
            C_t = max(b_p, ||T_t+c_t u_t||) T_t - max(0, b_p) (T_t+c_t u_t)

        with `u` being displacement jump increments, `t` denoting tangential component
        and `b_p` the friction bound.

        For `b_p = 0`, the equation `C_t = 0` does not in itself imply `T_t = 0`, which
        is what the contact conditions require. The case is handled through the use of a
        characteristic function.

        Parameters:
            fracture_subdomains: List of fracture subdomains.

        Returns:
            complementary_eq: Contact mechanics equation for the tangential constraints.

        """
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = (
            nd_vec_to_tangential
            @ self.contact_traction(subdomains)
            / self.characteristic_traction(subdomains)
        )
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Vectors needed to express the governing equations
        ones_frac = pp.ad.DenseArray(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        tol = self._characteristic_tolerance
        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)
        c_num = pp.ad.sum_operator_list(
            [e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis]
        )

        # Combine the above into expressions that enter the equation. c_num will
        # effectively be a sum of SparseArrays, thus we use a matrix-vector product @
        tangential_sum = t_t + c_num @ u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")
        bound = self.friction_bound(subdomains)
        b_p = self._max(subdomains, bound, zeros_frac)
        if self.params.get("adaptive_smoothing", False):
            rho = self.rho(subdomains)
            one = pp.ad.Scalar(1)

            def sigmoid(x):
                f_exp = pp.ad.Function(pp.ad.functions.exp, "exp_function")
                return one / (one + f_exp((pp.ad.Scalar(tol) - x) / rho))

            characteristic = scalar_to_tangential @ sigmoid(bound)

        else:
            characteristic: pp.ad.Operator = scalar_to_tangential @ (
                f_characteristic(b_p)
            )
        characteristic.set_name("characteristic_function_of_b_p")
        b_p.set_name("bp")

        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        # For the use of @, see previous comment.
        maxbp_abs = scalar_to_tangential @ self._max(
            subdomains, b_p, norm_tangential_sum
        )

        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        if self.params.get("relative_deformation_equations", False):
            equation /= (
                scalar_to_tangential
                @ self.characteristic_fracture_traction_operator(subdomains) ** 2
            )
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def characteristic_fracture_traction_estimate(self, subdomains):
        """Estimate the characteristic fracture traction.

        The characteristic fracture traction is estimated as the maximum of the
        contact traction over the fracture subdomains.

        Parameters:
            subdomains: List of subdomains where the contact traction is defined.

        Returns:
            Characteristic fracture traction.

        """
        # The normal component of the contact traction and the displacement jump
        t: pp.ad.Operator = self.contact_traction(
            subdomains
        ) / self.characteristic_traction(subdomains)
        e_n = self.e_i(subdomains, dim=self.nd, i=self.nd - 1)

        u = self.displacement_jump(subdomains) - e_n @ self.fracture_gap(subdomains)
        c_num = self.contact_mechanics_numerical_constant(subdomains)
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd), "norm_function")
        return f_norm(t) + f_norm(c_num * u)

    def characteristic_fracture_traction_value(self, subdomains):
        op = self.characteristic_fracture_traction_estimate(subdomains)
        val = self._compute_scalar_traction(op.value(self.equation_system))
        return val

    def characteristic_fracture_traction_operator(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        op = self.characteristic_fracture_traction_estimate(subdomains)

        def characteristic_traction(x):
            y = x if not isinstance(x, pp.ad.AdArray) else x.val
            val = self._compute_scalar_traction(y) * np.ones_like(y)
            if isinstance(x, pp.ad.AdArray):
                return pp.ad.AdArray(val, sps.csr_matrix(x.jac.shape))
            return val

        f_const = pp.ad.Function(characteristic_traction, "characteristic_traction")
        return f_const(op)

    def _compute_scalar_traction(self, val):
        val = val.clip(1e-8, 1e8)
        p = self.params.get("characteristic_traction_p", 5.0)
        p_mean = np.mean(val**p, axis=0) ** (1 / p)
        return p_mean


class ContactIndicators:
    @property
    def _characteristic_tolerance(self) -> float:
        return 1e-8

    def opening_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Function describing the state of the opening constraint.

        Negative for open fractures, positive for closed ones.

        Parameters:

        """
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump
        t_n: pp.ad.Operator = (
            nd_vec_to_normal
            @ self.contact_traction(subdomains)
            / self.characteristic_traction(subdomains)
        )
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)
        c_num = self.contact_mechanics_numerical_constant(subdomains)
        a = pp.ad.Scalar(-1.0) * t_n
        b = c_num * (u_n - self.fracture_gap(subdomains))
        ind = a - b
        if self.params.get("relative_violation", False):
            # Base on all fracture subdomains
            all_subdomains = self.mdg.subdomains(dim=self.nd - 1)
            ind = ind / pp.ad.Scalar(
                self.characteristic_fracture_traction_value(all_subdomains)
            )
        return ind

    def characteristic_traction(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        t_char = self.characteristic_displacement(subdomains) * self.youngs_modulus(
            subdomains
        )

        return t_char

    def characteristic_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        t_char = pp.ad.Scalar(
            self.solid.convert_units(1e-2, "m")
            * self.params.get("characteristic_displacement_scaling", 1.0),
            name="characteristic_displacement",
        )
        return t_char

    def sliding_indicator(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """
        Contact mechanics equation for the tangential constraints.

        The function reads
        .. math::
            C_t = max(b_p, ||T_t+c_t u_t||) T_t - max(0, b_p) (T_t+c_t u_t)

        with `u` being displacement jump increments, `t` denoting tangential component
        and `b_p` the friction bound.

        For `b_p = 0`, the equation `C_t = 0` does not in itself imply `T_t = 0`, which
        is what the contact conditions require. The case is handled through the use of a
        characteristic function.

        Negative for sticking, positive for sliding:  ||T_t+c_t u_t||-b_p

        Parameters:
            fracture_subdomains: List of fracture subdomains.

        Returns:
            complementary_eq: Contact mechanics equation for the tangential constraints.

        """

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )

        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = (
            nd_vec_to_tangential
            @ self.contact_traction(subdomains)
            / self.characteristic_traction(subdomains)
        )
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        # Functions EK: Should we try to agree on a name convention for ad functions?
        # EK: Yes. Suggestions?
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # With the active set method, the performance of the Newton solver is sensitive
        # to changes in state between sticking and sliding. To reduce the sensitivity to
        # round-off errors, we use a tolerance to allow for slight inaccuracies before
        # switching between the two cases.
        tol = self._characteristic_tolerance
        # The characteristic function will evaluate to 1 if the argument is less than
        # the tolerance, and 0 otherwise.
        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)

        b_p.set_name("bp")
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        c_num = pp.ad.sum_operator_list(
            [e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis]
        )
        tangential_sum = t_t + c_num @ u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        characteristic: pp.ad.Operator = scalar_to_tangential @ f_characteristic(b_p)
        characteristic.set_name("characteristic_function_of_b_p")
        f_heaviside = pp.ad.Function(heaviside, "heaviside_function")
        h_oi = f_heaviside(self.opening_indicator(subdomains))
        ind = norm_tangential_sum - b_p

        if self.params.get("relative_violation", False):
            # Base on all fracture subdomains
            all_subdomains = self.mdg.subdomains(dim=self.nd - 1)

            scale = self.characteristic_fracture_traction_value(all_subdomains)
            ind = ind / pp.ad.Scalar(scale)
        return ind * h_oi

    def friction_bound(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction bound [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction bound operator [Pa].

        """
        t_n: pp.ad.Operator = (
            self.normal_component(subdomains)
            @ self.contact_traction(subdomains)
            / self.characteristic_traction(subdomains)
        )
        u_n: pp.ad.Operator = self.normal_component(
            subdomains
        ) @ self.displacement_jump(subdomains)
        c_num = self.contact_mechanics_numerical_constant(subdomains)
        bound: pp.ad.Operator = self.friction_coefficient(subdomains) * (
            pp.ad.Scalar(-1.0) * t_n - c_num * (u_n - self.fracture_gap(subdomains))
        )
        bound.set_name("friction_bound")
        return bound


class TangentialKKT:
    def tangential_fracture_deformation_equation(self, subdomains):
        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Basis vectors for the tangential components. This is a list of Ad matrices,
        # each of which represents a cell-wise basis vector which is non-zero in one
        # dimension (and this is known to be in the tangential plane of the subdomains).
        # Ignore mypy complaint on unknown keyword argument
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )

        # To map a scalar to the tangential plane, we need to sum the basis vectors. The
        # individual basis functions have shape (Nc * (self.nd - 1), Nc), where Nc is
        # the total number of cells in the subdomain. The sum will have the same shape,
        # but the row corresponding to each cell will be non-zero in all rows
        # corresponding to the tangential basis vectors of this cell. EK: mypy insists
        # that the argument to sum should be a list of booleans. Ignore this error.
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )
        tangential_sum_matrix = pp.ad.sum_operator_list(
            [e_i.T for e_i in tangential_basis]
        )

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = (
            nd_vec_to_tangential
            @ self.contact_traction(subdomains)
            / self.characteristic_traction(subdomains)
        )
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Vectors needed to express the governing equations
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        # Functions EK: Should we try to agree on a name convention for ad functions?
        # EK: Yes. Suggestions?
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # The numerical constant is used to loosen the sensitivity in the transition
        # between sticking and sliding.
        # Expanding using only left multiplication to with scalar_to_tangential does not
        # work for an array, unlike the operators below. Arrays need right
        # multiplication as well.
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        # f = b - |t_t| >= 0
        # g = t_t \cdot u_t - b * |u_t| >= 0

        # Paper by Wietse Boon, 2019, An adaptive penalty method...
        # \Phi = f, \phi = -g
        # C = f - max(f - g, 0)
        b = self.friction_bound(subdomains)
        b_p = f_max(b, zeros_frac)
        f = b_p - f_norm(t_t)
        # First minus accounts for the anti in anti-parellel.
        g = tangential_sum_matrix @ (t_t * u_t_increment) - b_p * f_norm(u_t_increment)
        g *= c_num_as_scalar

        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        sharp_max = max_function(
            f - g,
            zeros_frac,
        )
        equation: pp.ad.Operator = f - sharp_max
        if self.params.get("adaptive_smoothing", False):

            a = f - g
            b = zeros_frac
            smoother = self.smooth_max(subdomains, sharp_max, a, b)
            equation += smoother

        # rho = pp.ad.Scalar(1e-3)
        # s = f + g
        # f_log = pp.ad.Function(pp.ad.functions.log, "log_function")
        # f_exp = pp.ad.Function(pp.ad.functions.exp, "exp_function")
        # smooth_max = s + rho * (f_log(pp.ad.Scalar(1) + f_exp(-s / rho)))
        # smooth_max = rho * (f_log(pp.ad.Scalar(1) + f_exp(-s / rho)))
        # equation: pp.ad.Operator = f - smooth_max
        tol = self._characteristic_tolerance

        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )  # zero if abs(x) < tol, 1 otherwise
        if self.nd == 3:
            e_0 = tangential_basis[0]
            e_1 = tangential_basis[1]
            # Add second equation, namely u_t \cdot t_t.T = 0
            # The dot product is u_t_0 * t_t_1 + u_t_1 * t_t_0

            eq_1 = (e_0.T @ u_t_increment) * (e_1.T @ t_t) - (  # e_1 @ (
                e_1.T @ u_t_increment
            ) * (e_0.T @ t_t)
            nz = f_norm(t_t) + f_norm(u_t_increment) * c_num_as_scalar
            chi = f_characteristic(nz)
            chi_open = f_characteristic(b_p)
            one = pp.ad.Scalar(1)
            chi_s = f_characteristic(f_norm(t_t) - b_p)
            chi_slip = (one - chi_open) * chi_s
            chi_stick = (one - chi_open) * (one - chi_s)
            whole = True
            if whole:
                equation = e_0 @ equation

                chi_open = scalar_to_tangential @ chi_open
                chi_slip = scalar_to_tangential @ chi_slip
                chi_stick = scalar_to_tangential @ chi_stick
                chi = scalar_to_tangential @ chi
                equation += e_1 @ (eq_1 * c_num_as_scalar)
                # equation = (one - chi) * equation + chi * t_t
                # equation = (
                #     chi_open * t_t
                #     + chi_stick * (c_num_as_scalar * u_t_increment)
                #     + chi_slip * equation
                # )
                # equation = (one - chi_open) * equation + chi_open * t_t
                equation = (one - chi) * equation + chi * t_t
            else:
                equation = (one - chi) * equation + chi * f_norm(t_t)

                equation = e_0 @ equation

                equation += e_1 @ (
                    (one - chi) * eq_1 * c_num_as_scalar + chi * f_norm(t_t)
                )

        equation.set_name("tangential_fracture_deformation_equation")
        return equation


class InteriorPoint:
    def normal_fracture_deformation_equation(self, subdomains):
        eq = super().normal_fracture_deformation_equation(subdomains)
        if self.params.get("adaptive_smoothing", False):
            name = eq.name
            eq += self.residual_variable(subdomains)
            eq.set_name(name)
        return eq

    def tangential_fracture_deformation_equation(self, subdomains):
        eq = super().tangential_fracture_deformation_equation(subdomains)
        if self.params.get("adaptive_smoothing", False):
            name = eq.name
            tangential_basis: list[pp.ad.SparseArray] = self.basis(
                subdomains, dim=self.nd - 1  # type: ignore[call-arg]
            )

            scalar_to_tangential = pp.ad.sum_operator_list(
                [e_i for e_i in tangential_basis]
            )
            eq += scalar_to_tangential @ self.residual_variable(subdomains)
            eq.set_name(name)
        return eq


class NonzeroInitialCondition:
    def initial_condition(self) -> None:
        """Set the initial condition for the problem."""
        super().initial_condition()
        for var in self.equation_system.variables:
            if hasattr(self, "initial_" + var.name):
                values = getattr(self, "initial_" + var.name)([var.domain])
                self.equation_system.set_variable_values(
                    values, [var], iterate_index=0, time_step_index=0
                )
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.set_solution_values(
                name="constraint_weights",
                values=np.ones(sd.num_cells),
                data=data,
                iterate_index=0,
            )

    def initial_pressure(self, sd=None):
        val = self.fluid.pressure()
        if sd is None:
            return val
        else:
            return val * np.ones(sd[0].num_cells)

    def initial_temperature(self, sd=None):
        val = self.fluid.temperature()
        if sd is None:
            return val
        else:
            return val * np.ones(sd[0].num_cells)

    def initial_t(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Initial contact traction [Pa].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for initial contact traction.

        """
        sd = subdomains[0]
        traction_vals = np.zeros((self.nd, sd.num_cells))
        traction_vals[-1] = -self.characteristic_traction(subdomains).value(
            self.equation_system
        ) / self.params.get("characteristic_displacement_scaling", 1.0)
        return traction_vals.ravel("F")


class Cnum:
    """Numerical constant for the contact problem [Pa * m^-1]."""

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        """Numerical constant for the contact problem [Pa * m^-1].

        Parameters:
            subdomains: List of subdomains. Only the first is used.

        Returns:
            c_num: Numerical constant, as scalar.

        """
        op = pp.ad.Scalar(1e0) / self.characteristic_displacement(subdomains)
        return op


class DynamicFriction:
    def friction_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction coefficient.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction coefficient operator.

        """
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        tol = 1e-10  # FIXME: Revisit this tolerance!
        # The characteristic function will evaluate to 1 if the argument is less than
        # the tolerance, and 0 otherwise.
        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )
        nd_vec_to_tangential = self.tangential_component(subdomains)

        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # We want two different values, depending on whether the tangential displacement
        # jump is zero or not. We use the characteristic function to switch between the
        # two values.
        op = pp.ad.Scalar(self.solid.friction_coefficient()) * (
            pp.ad.Scalar(0.75)
            + pp.ad.Scalar(0.25) * f_characteristic(f_norm(u_t_increment))
        )
        op.set_name("friction_coefficient")
        return op


class DarcysLawAd(pp.constitutive_laws.DarcysLawAd):
    # def set_geometry(self):
    #     """Set geometry for the model."""
    #     super().set_geometry()
    #     import meshplex

    #     for sd in self.mdg.subdomains(dim=2):

    #         mesh = meshplex.Mesh(sd.nodes[:].T, sd.tri.T)
    #         sd.cell_centers = mesh.cell_circumcenters.T
    #         # Compute face centers as intersections of circumcenter connections and edges.
    #         for f, cells in enumerate(sd.cell_face_as_dense().T):
    #             if -1 in cells:
    #                 continue
    #             nodes = sd.face_nodes[:, f].nonzero()[0]

    #             p = pp.intersections.segments_3d(
    #                 sd.cell_centers[:, cells[0]],
    #                 sd.cell_centers[:, cells[1]],
    #                 sd.nodes[:, nodes[0]],
    #                 sd.nodes[:, nodes[1]],
    #             )
    #             sd.face_centers[:, f] = p.flatten()
    #     # self.mdg.compute_geometry()

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        if all([sd.dim < self.nd for sd in subdomains]):
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return super().darcy_flux_discretization(subdomains)

    def fourier_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        if all([sd.dim < self.nd for sd in subdomains]):
            return pp.ad.TpfaAd(self.fourier_keyword, subdomains)
        else:
            return super().fourier_flux_discretization(subdomains)


def override_methods(
    cls,
    method_name: list[str],
    dofs: list[str, int],
    sort_criterion: Callable[[Any, pp.GridLike], bool] = None,
):

    if sort_criterion is None:
        sort_criterion = lambda g, cls: True

    def new_method(self, domains):
        super_method = getattr(super(cls, self), method_name)

        if len(domains) == 0 or all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            return super_method(domains)

        domains_h = [g for g in domains if sort_criterion(self, g)]
        domains_l = [g for g in domains if not sort_criterion(self, g)]
        proj = pp.ad.SubdomainProjections(domains, dofs[1])
        dof_type = dofs[0]
        # Check if dof_type value is plural (e.g. "faces"). If so, remove the last
        # character to get the singular form.
        if dof_type[-1] == "s":
            dof_type = dof_type[:-1]
        prol_h = getattr(proj, dof_type + "_prolongation")(domains_h)
        prol_l = getattr(proj, dof_type + "_prolongation")(domains_l)
        result = prol_h @ super_method(domains_h) + prol_l @ super_method(domains_l)
        return result

    setattr(cls, method_name, new_method)


class FixPrimalVariables:
    """Set a pressure/temperature condition in specified internal cells.

    This mimicks a pressure controlled well with known temperature and pressure.

    """

    def fracture_center_cell(self, sd: pp.Grid) -> np.ndarray:
        """Return the source term for the fracture."""
        mean_coo = np.mean(sd.cell_centers, axis=1).reshape((3, 1))
        center_cell = sd.closest_cell(mean_coo)
        val_loc = np.zeros(sd.num_cells)
        val_loc[center_cell] = 1
        return val_loc

    def in_out(self, sd: pp.Grid) -> np.ndarray:
        """Return the source term for the fracture."""
        return (-1) ** sd.frac_num

    def _well_cells(self, sd: pp.Grid) -> np.ndarray:
        """Return the pressure well cells."""
        if sd.dim == self.nd - 1:
            return self.fracture_center_cell(sd)
        else:
            return np.zeros(sd.num_cells)

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq = super().mass_balance_equation(subdomains)
        name = eq.name

        def well_pressure(sd):
            if sd.dim == self.nd - 1:
                if self.in_out(sd) > 0:
                    val = self.fluid.pressure() + 1.5e5
                else:
                    val = self.fluid.pressure() - 1.0e5
            else:
                val = 0
            return val * np.ones(sd.num_cells)

        eq = self._replace_well_rows(
            eq, self.pressure(subdomains), well_pressure, subdomains
        )
        eq.set_name(name)
        return eq

    def energy_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq = super().energy_balance_equation(subdomains)
        name = eq.name

        def well_temperature(sd):
            if sd.dim == self.nd - 1:
                if self.in_out(sd) > 0:
                    val = self.fluid.temperature() - self.params.get(
                        "inlet_temperature_difference", 10.0
                    )
                else:
                    val = self.fluid.temperature()
            else:
                val = 0
            return val * np.ones(sd.num_cells)

        eq = self._replace_well_rows(
            eq, self.temperature(subdomains), well_temperature, subdomains
        )
        eq.set_name(name)
        return eq

    def _replace_well_rows(
        self,
        eq: pp.ad.Operator,
        variable: pp.ad.Operator,
        values: np.ndarray | float | Callable[[pp.Grid], np.ndarray],
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Replace the rows corresponding to the well cells with the well value."""
        wells = np.hstack([self._well_cells(sd) for sd in subdomains])
        well_row_eliminator = pp.ad.SparseArray(sps.diags(1 - wells))
        if callable(values):
            values = np.hstack([values(sd) for sd in subdomains])
        well_values = pp.ad.DenseArray(values * wells)
        return (
            well_row_eliminator @ eq - well_values + pp.ad.DenseArray(wells) * variable
        )


class Common(
    # TangentialKKT,
    # AdaptiveSmoothing,
    ScaledContactMechanics,
    ContactIndicators,
    DisplacementBoundaryConditionsLinear,
    Cnum,
    NonzeroInitialCondition,
    IterationExporting,
    ExportScaledData,
    pp.DiagnosticsMixin,
):
    """Common functionality for the models."""


class MechanicsModel(
    # DynamicFriction,
    pp.constitutive_laws.DisplacementJumpAperture,
    Common,
    pp.momentum_balance.MomentumBalance,
):
    """Mixed-dimensional elastic problem with dynamic friction."""


class PoromechanicsModel(
    FluidFlowBoundaryConditionsDirEastWest,
    DarcysLawAd,
    pp.constitutive_laws.CubicLawPermeability,
    Common,
    pp.poromechanics.Poromechanics,
):
    """Mixed-dimensional poroelastic problem."""


class ThermoporomechanicsModel(
    EnergyBoundaryConditionsDirEastWest,
    FluidFlowBoundaryConditionsDirEastWest,
    DarcysLawAd,
    pp.constitutive_laws.FouriersLawAd,
    pp.constitutive_laws.CubicLawPermeability,
    Common,
    pp.thermoporomechanics.Thermoporomechanics,
):
    """Mixed-dimensional poroelastic problem."""


def physical_models(smoothing=None):
    bases = [PoromechanicsModel, ThermoporomechanicsModel]
    names = ["Poromechanics", "Thermoporomechanics"]
    models = []
    for b, n in zip(bases, names):
        if smoothing == "explicit":

            class Model(AdaptiveSmoothingExplicit, b):
                pass

        elif smoothing == "implicit":

            class Model(
                AdaptiveSmoothingImplicit,
                # InteriorPoint,
                b,
            ):
                pass

        else:

            class Model(b):
                pass

        models.append((n, Model))
    return models
