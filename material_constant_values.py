import porepy as pp

granite_values: dict[str, float] = pp.solid_values.granite
granite_values.update(
    {
        # "biot_coefficient": 0.8,
        "permeability": 2e-15,
        "normal_permeability": 1e-7,
        # "porosity": 1.0e-2,
        # "shear_modulus": 16.67 * pp.GIGA,
        # "lame_lambda": 11.11 * pp.GIGA,
        # "specific_heat_capacity": 7.9e2,
        # "thermal_conductivity": 2.5,
        # "thermal_expansion": 1e-4,
        "residual_aperture": 2e-5,
        # "density": 2.7e3,
        "maximum_fracture_closure": 0e-4,
        "fracture_normal_stiffness": 10 * pp.GIGA,
        "fracture_gap": 0e-4,  # Equals the maximum fracture closure.
        "dilation_angle": 0.1,
        "friction_coefficient": 1.0,
    }
)


# Cf. fluid.py
water_values = pp.fluid_values.water
reference_temperature = 0.0
fluid_values: dict[str, float] = {
    "compressibility": 1e-6,
    "viscosity": 1e-1,
    "density": 1.0e0,
    "thermal_conductivity": 1,
    "thermal_expansion": 1e-2,
    "specific_heat_capacity": 1e2,
    "temperature": reference_temperature,
}
plain_rock_values: dict[str, float] = {
    "biot_coefficient": 0.8,
    "permeability": 1e-8,
    "normal_permeability": 1e-6,
    "porosity": 1.0e-2,
    "shear_modulus": 2e6,
    "lame_lambda": 2e6,
    "specific_heat_capacity": 1e2,
    "thermal_conductivity": 1.0,
    "thermal_expansion": 1e-3,
    "residual_aperture": 1e-3,
    "density": 1e0,
    "maximum_fracture_closure": 0e-3,
    "fracture_normal_stiffness": 1e3,
    "fracture_gap": 0e-3,  # Equals the maximum fracture closure.
    "dilation_angle": 0.2,
    "friction_coefficient": 1.0,
    "temperature": reference_temperature,
}
