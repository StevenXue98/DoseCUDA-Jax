"""Fast tests for the repository's array/physical-axis contract."""

import os
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from DoseCUDA import IMPTBeam, IMPTDoseGrid


class IndexingContractTests(unittest.TestCase):
    def test_noncubic_phantom_maps_zyx_shape_to_xyz_origin(self):
        grid = IMPTDoseGrid()
        grid.createCubePhantom(size=(13, 15, 17), spacing=2.0)

        self.assertEqual(grid.HU.shape, (13, 15, 17))
        self.assertEqual(tuple(grid.size), (13, 15, 17))
        np.testing.assert_array_equal(grid.spacing, (2.0, 2.0, 2.0))
        np.testing.assert_array_equal(grid.origin, (-17.0, -15.0, -13.0))

    def test_simpleitk_round_trip_preserves_axes_and_metadata(self):
        grid = IMPTDoseGrid()
        grid.HU = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        grid.size = np.array(grid.HU.shape)
        grid.origin = np.array((11.0, 22.0, 33.0), dtype=np.float32)
        grid.spacing = np.array((1.5, 2.5, 3.5), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "axis-coded.nrrd")
            grid.writeCTNRRD(path)

            image = sitk.ReadImage(path)
            np.testing.assert_array_equal(sitk.GetArrayFromImage(image), grid.HU)
            self.assertEqual(image.GetSize(), (5, 4, 3))
            np.testing.assert_allclose(image.GetOrigin(), grid.origin)
            np.testing.assert_allclose(image.GetSpacing(), grid.spacing)

            loaded = IMPTDoseGrid()
            loaded.loadCTNRRD(path)
            np.testing.assert_array_equal(loaded.HU, grid.HU)
            self.assertEqual(tuple(loaded.size), grid.HU.shape)

    def test_one_spot_keeps_a_two_dimensional_spot_table(self):
        beam = IMPTBeam()
        beam.addSingleSpot(12.0, -7.0, 0.5, 23)

        self.assertEqual(beam.spot_list.shape, (1, 4))
        np.testing.assert_array_equal(
            beam.spot_list[0], np.array((12.0, -7.0, 0.5, 23.0), dtype=np.float32)
        )

    def test_inconsistent_size_is_rejected(self):
        grid = IMPTDoseGrid()
        grid.HU = np.zeros((3, 4, 5), dtype=np.float32)
        grid.size = np.array((5, 4, 3))
        grid.origin = np.zeros(3, dtype=np.float32)
        grid.spacing = np.ones(3, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "size must match HU.shape"):
            grid._validate_geometry()


if __name__ == "__main__":
    unittest.main()
