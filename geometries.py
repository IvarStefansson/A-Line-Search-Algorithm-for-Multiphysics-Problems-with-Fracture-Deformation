# %%
import logging

import numpy as np
import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures, SquareDomainOrthogonalFractures)

logger = logging.getLogger(__name__)
# %%


class _RandomFractures2d:
    """Model geometry mixin for creating random fractures."""

    _nd: int = 2

    def set_fractures(self) -> None:
        num_fractures = self.params.get("num_fractures", 1)
        self._fractures = []
        index, counter = 0, 0
        while counter < num_fractures:
            frac = self.random_fracture(index)
            index += 1
            if self.accept_new_fracture(frac):
                self._fractures.append(frac)
                counter += 1

    def accept_new_fracture(self, frac: pp.LineFracture) -> bool:
        """Check if a new fracture is accepted.

        Parameters:
            frac: Fracture to be checked.

        Returns:
            True if the fracture is accepted, False otherwise.

        """
        # Check if the fracture is within the domain. We only want fractures that are fully within the domain.
        accept = True
        tolerance = 1e-2
        for i in range(self._nd):
            if np.any(frac.pts[i] < tolerance) or np.any(
                frac.pts[i] > self.domain_size - tolerance
            ):
                accept = False
        return accept

    def random_fracture(self, i: int) -> pp.LineFracture:
        """Return a random fracture."""
        np.random.seed(i)
        center = self.fracture_center(i)[: self._nd]
        point_1 = center + self.fracture_size(i) / 2 * np.array(
            [np.cos(self.fracture_angle(i)), np.sin(self.fracture_angle(i))]
        ).reshape(center.shape)
        # Fracture extends by a length fracture_size in a random direction from point_1
        point_2 = point_1 - self.fracture_size(i) / 2 * np.array(
            [np.cos(self.fracture_angle(i)), np.sin(self.fracture_angle(i))]
        ).reshape(center.shape)
        pts = np.hstack([point_1, point_2])
        return pp.LineFracture(pts, index=i)

    def fracture_center(self, i: int) -> np.ndarray:
        """Return the center of fracture i.

        Parameters:
            i: Index of fracture.

        Returns:
            Center of fracture.

        """
        coo = np.zeros(3)
        np.random.seed(i)
        coo[: self._nd] = np.random.rand(self._nd) * self.domain_size
        return coo.reshape((3, 1))

    def fracture_size(self, i: int) -> float:
        """Return the size of fracture i.

        Could be perturbed by a random factor.

        Parameters:
            i: Index of fracture.

        Returns:
            Size of fracture.

        """

        custom = self.params.get("fracture_size")
        if custom is not None:
            if isinstance(custom, list):
                size = self.solid.convert_units(custom[i], "m")
            else:
                size = self.solid.convert_units(custom, "m")
        else:
            size = 0.2 * self.domain_size
        return size

    def fracture_angle(self, i: int) -> float:
        """Return the angle of fracture i.

        Parameters:
            i: Index of fracture.

        Returns:
            Angle of fracture.

        """
        np.random.seed(i)
        return np.random.rand() * np.pi * 2

    def grid_type(self) -> str:
        return "simplex"


class RandomFractures2d(_RandomFractures2d, SquareDomainOrthogonalFractures):
    """Model geometry containing random fractures in a square domain."""

    pass


# %%
class _RandomFractures3d(_RandomFractures2d):
    """Model geometry mixin for creating random elliptic fractures."""

    _nd: int = 3

    def random_fracture(self, i: int) -> pp.PlaneFracture:
        """Return a random fracture."""
        np.random.seed(i)
        frac = pp.create_elliptic_fracture(
            center=self.fracture_center(i),
            major_axis=self.fracture_size(i) / 2,
            minor_axis=self.fracture_size(i) / 2,
            major_axis_angle=self.fracture_angle(i),
            strike_angle=self.strike_angle(i),
            dip_angle=self.dip_angle(i),
            num_points=self.params.get("num_fracture_points", 8),
            index=i,
        )
        return frac

    def strike_angle(self, i: int) -> float:
        """Return the strike angle of fracture i.

        Parameters:
            i: Index of fracture.

        Returns:
            Strike angle of fracture.

        """
        np.random.seed(i + 1)
        return np.random.rand() * np.pi * 2

    def dip_angle(self, i: int) -> float:
        """Return the dip angle of fracture i.

        Parameters:
            i: Index of fracture.

        Returns:
            Dip angle of fracture.

        """
        np.random.seed(i + 2)
        return np.random.rand() * np.pi * 2


class RandomFractures3d(_RandomFractures3d, CubeDomainOrthogonalFractures):
    """Model geometry containing random elliptic plane fractures in a cube domain."""

    pass
