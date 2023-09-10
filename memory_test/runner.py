import json
import subprocess


class Runner:
    exePath = "C://development//optix//OptixRCS//build//bin//Release//ExtremeSpeedBranch.exe"

    is_debug = False
    center_relocate = False

    rays_per_wavelength = 10
    trace_depth = 10

    freq = 3E9
    reflectance = 1.0

    phi_start = 0
    phi_end = 90
    phi_interval = 5

    theta_start = 90
    theta_end = 90
    theta_interval = 1

    polarization = "HH"

    model_file = "C:/development/optix/OptixRCS/resources/corner_reflector.obj"

    csv_file = "C:/development/optix/OptixRCS/output/test.csv"

    def generate_json(self):
        # Dictionary that will be written to the JSON config file
        config = {
            "is_debug": self.is_debug,
            "center_relocate": self.center_relocate,
            "rays_per_wavelength": self.rays_per_wavelength,
            "trace_depth": self.trace_depth,
            "freq": self.freq,
            "reflectance": self.reflectance,
            "phi_start": self.phi_start,
            "phi_end": self.phi_end,
            "phi_interval": self.phi_interval,
            "theta_start": self.theta_start,
            "theta_end": self.theta_end,
            "theta_interval": self.theta_interval,
            "polarization": self.polarization,
            "model_file": self.model_file,
            "csv_file": self.csv_file
        }

        # Write to the JSON config file
        with open('C:/development/optix/OptixRCS/test/config.json', 'w') as f:
            # indent parameter is optional, it makes the file human-readable
            json.dump(config, f, indent=4)

    def run(self):
        self.generate_json()
        config_path = "C:/development/optix/OptixRCS/test/config.json"
        subprocess.run([self.exePath, config_path])


def main():
    runner = Runner()
    model_name = "x35"
    model_file = "C://development//optix//OptixRCS//resources//" + model_name+".obj"
    csv_file_vs = "C://development//optix//OptixRCS//time_test//" + 'x35_time'

    runner.model_file = model_file
    runner.csv_file = csv_file_vs
    runner.freq = 5E8
    runner.center_relocate = False
    runner.phi_start = 0
    runner.phi_end = 90
    runner.phi_interval = 1
    runner.theta_start = 90
    runner.theta_end = 90
    runner.rays_per_wavelength = 380
    runner.trace_depth = 10
    runner.generate_json()
    runner.run()


if __name__ == "__main__":
    main()
