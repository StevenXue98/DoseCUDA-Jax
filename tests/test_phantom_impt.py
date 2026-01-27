import os
import numpy as np
from DoseCUDA import IMPTDoseGrid, IMPTPlan, IMPTBeam

#check if test_phantom_output directory exists
if not os.path.exists("test_phantom_output"):
    os.makedirs("test_phantom_output")

# create dose grid, plan, and beam objects
dose = IMPTDoseGrid()
plan = IMPTPlan()
beam = IMPTBeam()

# initialize the default digital cube phantom
dose.createCubePhantom()

# define a spot list - create a circle of spots incrementing the energy id
beam.dicom_rangeshifter_label = '0'
n_spots = 98
for energy_id in range(n_spots):
    theta = 2.0 * 3.14159 * energy_id / n_spots
    spot_x = 100.0 * np.cos(theta)
    spot_y = 100.0 * np.sin(theta)
    mu = 0.2
    beam.addSingleSpot(spot_x, spot_y, mu, energy_id)

# add the beam to the plan
plan.addBeam(beam)

# compute the dose
dose.computeIMPTPlan(plan)

# write the dose to a file
dose.writeDoseNRRD("test_phantom_output/cube_impt_dose.nrrd", individual_beams=False, dose_type="PHYSICAL")

# write the CT to a file
dose.writeCTNRRD("test_phantom_output/cube_phantom_ct.nrrd")

print("Done.")