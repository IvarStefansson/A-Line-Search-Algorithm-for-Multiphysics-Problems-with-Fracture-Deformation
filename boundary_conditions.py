from typing import Callable

import numpy as np
import porepy as pp


class DisplacementBoundaryConditionsShear:
    solid: pp.SolidConstants

    nd: int

    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions.

        Parameters:
            sd: Subdomain grid.

        Returns:
            bc: Boundary condition representation.

        """
        # Define boundary faces.
        boundary_faces = self.displacement_boundary_faces(sd)
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        # Default internal BC is Neumann. We change to Dirichlet, i.e., the
        # mortar variable represents the displacement on the fracture faces.
        bc.internal_to_dirichlet(sd)
        return bc

    def displacement_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define boundary faces.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Array of boundary faces.

        """
        boundary_faces = self.domain_boundary_sides(sd)
        if self.nd == 2:
            return boundary_faces.south + boundary_faces.north
        else:
            return boundary_faces.bottom + boundary_faces.top


class DisplacementBoundaryConditionsLinear(DisplacementBoundaryConditionsShear):
    time_manager: pp.TimeManager

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for mechanics.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """

        # Default is zero.
        vals = np.zeros((self.nd, boundary_grid.num_cells))
        if boundary_grid.dim < self.nd - 1:
            return vals.ravel("F")
        boundary_faces = self.domain_boundary_sides(boundary_grid)

        # Set Neumann conditions for force on the fracture faces.
        # Compressive stresses. Maximum horizontal along y-axis (east and west).
        if self.nd == 2:
            faces = boundary_faces.north
        else:
            faces = boundary_faces.top
        if self.time_manager.time < 1e-5:
            vals[-1, faces] = self.solid.residual_aperture()
            return vals.ravel("F")

        u_char = self.characteristic_displacement([boundary_grid]).value(
            self.equation_system
        ) / self.params.get("characteristic_displacement_scaling", 1.0)
        # Normal component of boundary displacement.
        axis = 0
        coo = (
            boundary_grid.cell_centers[axis, faces]
            + boundary_grid.cell_centers[1, faces]
        ) / np.max(boundary_grid.cell_centers[axis])
        # Produce different contact regimes along the fracture.
        linear_increase = u_char * (coo - 0.5)
        offset = -0.5 * u_char
        vals[-1, faces] = (
            self.solid.convert_units(offset - 2.0 * linear_increase, "m")
            + self.solid.fracture_gap()
        )
        # Tangential component of boundary displacement.
        if self.nd == 3:
            w = self.params.get("dir_u_weight", 3)
            vals[0, faces] = self.solid.convert_units(w * u_char * 2.0, "m")
            vals[1, faces] = self.solid.convert_units(w * u_char * 1.0, "m")
        else:
            w = self.params.get("dir_u_weight", 1.0)
            vals[0, faces] = self.solid.convert_units(w * u_char, "m")
        return vals.ravel("F")

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Stress values for the Dirichlet boundary condition.

        Parameters:
            boundary_grid: Boundary grid to evaluate values on.

        Returns:
            An array with shape (boundary_grid.num_cells,) containing the stress values
            on the provided boundary grid.

        """
        vals = np.zeros((self.nd, boundary_grid.num_cells))
        if self.time_manager.time > 1e-5 and boundary_grid.dim == self.nd - 1:
            dbs = self.domain_boundary_sides(boundary_grid)

            # Set Neumann conditions for force on the fracture faces.
            # Compressive stresses. Maximum horizontal along y-axis (east and west).
            val = (
                self.characteristic_displacement([boundary_grid]).value(
                    self.equation_system
                )
                / self.params.get("characteristic_displacement_scaling", 1.0)
                * self.solid.shear_modulus()
            )
            vals[0, dbs.east] = -val * boundary_grid.cell_volumes[dbs.east]
            vals[0, dbs.west] = val * boundary_grid.cell_volumes[dbs.west]
            if self.nd == 3:
                vals[1, dbs.north] = -val * boundary_grid.cell_volumes[dbs.north]
                vals[1, dbs.south] = val * boundary_grid.cell_volumes[dbs.south]

        return vals.ravel("F")


class FluidFlowBoundaryConditionsDirEastWest:
    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]
    is_well: Callable[[pp.Grid], bool]
    fluid: pp.FluidConstants

    @property
    def onset(self) -> bool:
        """Onset time at which we change boundary values at Fracture 1."""
        return self.time_manager.time > self.time_manager.schedule[0] + 1e-5

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        pressure = self.fluid.pressure() * np.ones(boundary_grid.num_cells)
        if boundary_grid.dim < self.nd - 1 and self.onset:
            sides = self.domain_boundary_sides(boundary_grid)
            pressure[sides.west] = self.fluid.convert_units(1e6, "Pa")
        return pressure

    def _dirichlet_darcy_faces(self, sd: pp.Grid) -> np.ndarray:
        """Find the faces on the Dirichlet boundary.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            faces: Array of faces on the Dirichlet boundary.

        """
        if sd.dim < self.nd:
            domain_sides = self.domain_boundary_sides(sd)
            faces = domain_sides.east  # + domain_sides.west
            return faces
        else:
            return np.array([], dtype=int)

    def _dirichlet_flux_faces(self, sd: pp.Grid) -> np.ndarray:
        """Find the faces on the Dirichlet boundary.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            faces: Array of faces on the Dirichlet boundary.

        """
        if sd.dim < self.nd:
            domain_sides = self.domain_boundary_sides(sd)
            faces = domain_sides.east + domain_sides.west
            return faces
        else:
            return np.array([], dtype=int)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return pp.BoundaryCondition(sd, self._dirichlet_darcy_faces(sd), "dir")

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(boundary_grid.num_cells)
        if boundary_grid.dim < self.nd - 1 and self.onset:
            sides = self.domain_boundary_sides(boundary_grid)
            # Currently pressure controlled.
            q_in = self.q_in
            vals[sides.west] = (
                self.fluid.convert_units(q_in, "m^2*s^-1")
                * boundary_grid.cell_volumes[sides.west]
            )
        return vals

    @property
    def q_in(self) -> float:
        """Inflow rate at the east boundary."""
        return self.fluid.convert_units(self.params.get("q_in", 0), "m^2*s^-1")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self._dirichlet_flux_faces(sd), "dir")


class EnergyBoundaryConditionsDirEastWest:
    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for enthalpy.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return pp.BoundaryCondition(sd, self._dirichlet_flux_faces(sd), "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Fourier flux.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return pp.BoundaryCondition(sd, self._dirichlet_flux_faces(sd), "dir")

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        temperature = self.fluid.temperature() * np.ones(boundary_grid.num_cells)
        if boundary_grid.dim < self.nd - 1 and self.onset:
            sides = self.domain_boundary_sides(boundary_grid)
            # T on boundary = T0-dT = T0 - (T0-T_in)
            temperature[sides.west] -= self.fluid.convert_units(
                self.params.get("inlet_temperature_difference", 10.0), "K"
            )
        return temperature


class PureCompression:
    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for mechanics.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """

        # Default is zero.
        vals = np.zeros((self.nd, boundary_grid.num_cells))
        if boundary_grid.dim < self.nd - 1:
            return vals.ravel("F")
        boundary_faces = self.domain_boundary_sides(boundary_grid)

        if self.nd == 2:
            faces = boundary_faces.north
        else:
            faces = boundary_faces.top
        if self.time_manager.time < 1e-5:
            return vals.ravel("F")
        u_char = self.characteristic_displacement([boundary_grid]).value(
            self.equation_system
        ) / self.params.get("characteristic_displacement_scaling", 1.0)
        # Normal component of boundary displacement.
        vals[-1, faces] = -u_char
        return vals.ravel("F")


class NeumannBoundaryConditions:

    def _dirichlet_flux_faces(self, sd: pp.Grid) -> np.ndarray:
        """Find the faces on the Dirichlet boundary.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            faces: Array of faces on the Dirichlet boundary.

        """
        if sd.dim == self.nd:
            domain_sides = self.domain_boundary_sides(sd)
            faces = domain_sides.all_bf
        else:
            faces = np.array([], dtype=int)
        return faces
