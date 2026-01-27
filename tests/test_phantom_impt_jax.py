import os
import sys
import numpy as np
import SimpleITK as sitk

# Add Jax folder to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
jax_dir = os.path.join(os.path.dirname(script_dir), "DoseCUDA", "Jax")
sys.path.insert(0, jax_dir)

from DoseCUDA import IMPTDoseGrid, IMPTPlan, IMPTBeam
from jax_impt import computeIMPTPlanJax

# check if test_phantom_output directory exists
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

# compute the dose using JAX
dose_jax = computeIMPTPlanJax(dose, plan)

# write the JAX dose to a file
dose_img = sitk.GetImageFromArray(dose_jax.astype(np.float32))
dose_img.SetOrigin(dose.origin)
dose_img.SetSpacing(dose.spacing)
sitk.WriteImage(dose_img, "test_phantom_output/cube_impt_dose_jax.nrrd")

# write the CT to a file
dose.writeCTNRRD("test_phantom_output/cube_phantom_ct_jax.nrrd")

print("Done.")
