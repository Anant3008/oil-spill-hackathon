import os
import logging
from datetime import datetime, timedelta
from opendrift.models.openoil import OpenOil

logger = logging.getLogger(__name__)

class SpillModel:
    """
    OpenDrift / OpenOil Execution Wrapper.
    Handles configuration, seeding, and execution of the trajectory model.
    The environment is completely decoupled and provided via an EnvironmentManager.
    """
    def __init__(self, start_lat, start_lon, start_time, oil_type='GENERIC HEAVY CRUDE'):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.start_time = start_time
        self.oil_type = oil_type
        
        # Initialize OpenOil model
        self.o = OpenOil(loglevel=30)
        
    def add_environment_readers(self, readers):
        """Adds pre-configured OpenDrift environmental readers."""
        for r in readers:
            self.o.add_reader(r)
            
    def run_simulation(self, duration_hours=24, time_step_hours=1, num_particles=1000, radius_m=5000, outfile="output/spill_trajectory.nc"):
        """Seeds particles and executes the physics simulation."""
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        logger.info(f"Seeding {num_particles} particles at ({self.start_lat}, {self.start_lon})")
        self.o.seed_elements(
            lon=self.start_lon,
            lat=self.start_lat,
            radius=radius_m,
            number=num_particles,
            time=self.start_time,
            oil_type=self.oil_type
        )

        logger.info(f"Running simulation for {duration_hours} hours...")
        self.o.run(
            duration=timedelta(hours=duration_hours),
            time_step=timedelta(hours=time_step_hours),
            outfile=outfile
        )
        
        logger.info(f"Simulation complete. Trajectory saved to {outfile}")
        return outfile
