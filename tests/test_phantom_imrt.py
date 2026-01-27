import os
import numpy as np
from DoseCUDA import IMRTDoseGrid, IMRTPlan, IMRTBeam

#check if test_phantom_output directory exists
if not os.path.exists("test_phantom_output"):
    os.makedirs("test_phantom_output")

# create dose grid, plan, and beam objects
dose = IMRTDoseGrid()
plan = IMRTPlan("VarianTrueBeamHF")

# initialize the default digital cube phantom
dose.createCubePhantom()

# define beam parameters
plan.addSquareField()
plan.addSquareField('6', 15.0, 15.0, mu=50, gantry_angle=90.0, collimator_angle=20.0)

# compute the dose
dose.computeIMRTPlan(plan)

# write the dose to a file
dose.writeDoseNRRD("test_phantom_output/cube_imrt_dose.nrrd", individual_beams=False, dose_type="PHYSICAL")

# write the CT to a file
dose.writeCTNRRD("test_phantom_output/cube_phantom_ct.nrrd")

print("Done.")