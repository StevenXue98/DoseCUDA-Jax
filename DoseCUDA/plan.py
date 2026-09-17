import numpy as np
import SimpleITK as sitk
import pydicom as pyd


class VolumeObject:

    def __init__(self):
        self.origin = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.spacing = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.voxel_data = []


class Prescription:

    def __init__(self):
        self.TargetPrescriptionDose = 0.0
        self.ROIName = None
        self.TargetUnderdoseVolumeFraction = 0.0


class DoseGrid:
    """Voxel grid with NumPy arrays in ``(z, y, x)`` axis order.

    ``origin`` and ``spacing`` follow the SimpleITK/DICOM physical coordinate
    convention and are therefore ordered ``(x, y, z)``.  ``size`` is always
    the NumPy array shape, not the SimpleITK image size.
    """

    def __init__(self):
        self.origin = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.spacing = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.size = np.array([0, 0, 0])
        self.HU = []
        self.dose = []
        self.beam_doses = []
        self.FrameOfReferenceUID = ""

    def _validate_geometry(self):
        """Reject inconsistent grid metadata before entering a dose engine."""
        hu = np.asarray(self.HU)
        if hu.ndim != 3:
            raise ValueError("HU must be a 3-dimensional array in (z, y, x) order")

        size = tuple(int(value) for value in np.asarray(self.size).tolist())
        if size != hu.shape:
            raise ValueError(
                f"size must match HU.shape in (z, y, x) order; got size={size} "
                f"and HU.shape={hu.shape}"
            )

        origin = np.asarray(self.origin)
        spacing = np.asarray(self.spacing)
        if origin.shape != (3,):
            raise ValueError("origin must contain three values in (x, y, z) order")
        if spacing.shape != (3,):
            raise ValueError("spacing must contain three values in (x, y, z) order")
        if not np.all(np.isfinite(origin)):
            raise ValueError("origin values must be finite")
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
            raise ValueError("spacing values must be finite and positive")

    def loadCTNRRD(self, ct_path):
        fr = sitk.ImageFileReader()
        fr.SetFileName(ct_path)
        ct_img = fr.Execute()

        self.origin = np.array(ct_img.GetOrigin())
        self.spacing = np.array(ct_img.GetSpacing())
        self.HU = np.array(sitk.GetArrayFromImage(ct_img), dtype=np.single)
        self.size = np.array(self.HU.shape)

    def loadCTDCM(self, ct_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ct_path)

        dicom_names = list(dicom_names)
        dicom_names.sort(key=lambda x: pyd.dcmread(x, force=True).ImagePositionPatient[2])

        reader.SetFileNames(dicom_names)
        ct_img = reader.Execute()

        self.origin = np.array(ct_img.GetOrigin())
        self.spacing = np.array(ct_img.GetSpacing())
        self.HU = np.array(sitk.GetArrayFromImage(ct_img), dtype=np.single)
        self.HU = np.clip(self.HU, -1000.0, None)
        self.size = np.array(self.HU.shape)

    def resampleCT(self, new_spacing, new_size, new_origin):
        """Resample using SimpleITK-style ``(x, y, z)`` metadata arguments."""
        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(np.asarray(self.origin, dtype=float).tolist())
        HU_img.SetSpacing(np.asarray(self.spacing, dtype=float).tolist())

        rf = sitk.ResampleImageFilter()
        rf.SetOutputOrigin(np.asarray(new_origin, dtype=float).tolist())
        rf.SetOutputSpacing(np.asarray(new_spacing, dtype=float).tolist())
        rf.SetSize(new_size)
        rf.SetDefaultPixelValue(-1000)

        HU_resampled = rf.Execute(HU_img)
        self.HU = np.array(sitk.GetArrayFromImage(HU_resampled), dtype=np.single)

        self.origin = new_origin
        self.spacing = new_spacing
        self.size = np.array(self.HU.shape)

    def resampleCTfromSpacing(self, spacing):

        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(np.asarray(self.origin, dtype=float).tolist())
        HU_img.SetSpacing(np.asarray(self.spacing, dtype=float).tolist())

        rf = sitk.ResampleImageFilter()
        rf.SetOutputOrigin(np.asarray(self.origin, dtype=float).tolist())
        sp_new = (spacing, spacing, spacing)
        sz_new = (int(self.size[2] * self.spacing[0] / sp_new[0]),
                  int(self.size[1] * self.spacing[1] / sp_new[1]),
                  int(self.size[0] * self.spacing[2] / sp_new[2]))
        rf.SetOutputSpacing(sp_new)
        rf.SetSize(sz_new)
        rf.SetDefaultPixelValue(-1000)

        HU_resampled = rf.Execute(HU_img)
        self.HU = np.array(sitk.GetArrayFromImage(HU_resampled), dtype=np.single)
        
        self.spacing = sp_new
        self.size = np.array(self.HU.shape)

    def resampleCTfromReferenceDose(self, ref_dose_path):

        ref_dose = pyd.dcmread(ref_dose_path, force=True)
        slice_thickness = float(ref_dose.GridFrameOffsetVector[1]) - float(ref_dose.GridFrameOffsetVector[0])
        ref_spacing = np.array([float(ref_dose.PixelSpacing[0]), float(ref_dose.PixelSpacing[1]), slice_thickness])
        ref_origin = np.array(ref_dose.ImagePositionPatient)

        ref_dose_img = sitk.GetImageFromArray(ref_dose.pixel_array)
        ref_dose_img.SetOrigin(np.asarray(ref_origin, dtype=float).tolist())
        ref_dose_img.SetSpacing(np.asarray(ref_spacing, dtype=float).tolist())

        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(np.asarray(self.origin, dtype=float).tolist())
        HU_img.SetSpacing(np.asarray(self.spacing, dtype=float).tolist())

        rf = sitk.ResampleImageFilter()
        rf.SetReferenceImage(ref_dose_img)
        rf.SetDefaultPixelValue(-1000)

        HU_resampled = rf.Execute(HU_img)

        self.HU = np.array(sitk.GetArrayFromImage(HU_resampled), dtype=np.single)

        self.size = np.array(self.HU.shape)
        self.origin = ref_origin
        self.spacing = ref_spacing

    def applyCouchModel(self, couch_wet=8.0):
        spacing = self.spacing[0]
        n_voxels = int(50.0 / spacing)
        hu_override_value = ((couch_wet / (n_voxels * spacing)) - 1.0) * 1000.0

        self.HU[:, -n_voxels:, :] = hu_override_value

    def writeDoseDCM(self, dose_path, ref_dose_path, dose_type="EFFECTIVE", individual_beams=False):

        if not dose_path.endswith(".dcm"):
            raise Exception("Dose path must have .dcm extension")
        else:
            print("test")

        if dose_type == "EFFECTIVE":
            RBE = 1.1
        elif dose_type == "PHYSICAL":
            RBE = 1.0
        else:
            raise Exception("Unknown dose type: %s" % dose_type)

        if individual_beams:
            for i, beam_dose in enumerate(self.beam_doses):
                ref_dose = pyd.dcmread(ref_dose_path, force=True)
                ref_dose.SeriesDescription = ref_dose.SeriesDescription + "_DoseCUDA"
                ref_dose.DoseSummationType = "BEAM"
                ref_dose.DoseType = dose_type
                ref_dose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = pyd.uid.generate_uid()

                dose_dcm = np.array(beam_dose * 500.0 * RBE, dtype=np.uint16)
                ref_dose.PixelData = dose_dcm
                ref_dose.DoseGridScaling = 0.002

                ref_dose.save_as(dose_path.replace(".dcm", "_beam%02i.dcm" % (i+1)))
        else:
            ref_dose = pyd.dcmread(ref_dose_path, force=True)
            ref_dose.SeriesDescription = ref_dose.SeriesDescription + "_DoseCUDA"
            ref_dose.DoseSummationType = "PLAN"
            ref_dose.DoseType = dose_type
            ref_dose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = pyd.uid.generate_uid()
            
            dose_dcm = np.array(self.dose * 500.0 * RBE, dtype=np.uint16)
            ref_dose.PixelData = dose_dcm
            ref_dose.DoseGridScaling = 0.002

            ref_dose.save_as(dose_path)

    def writeDoseNRRD(self, dose_path, individual_beams=False, dose_type="EFFECTIVE"):

        if not dose_path.endswith(".nrrd"):
            raise Exception("Dose path must have .nrrd extension")
        
        if dose_type == "EFFECTIVE":
            RBE = 1.1
        elif dose_type == "PHYSICAL":
            RBE = 1.0
        else:
            raise Exception("Unknown dose type: %s" % dose_type)

        fw = sitk.ImageFileWriter()
        dose_img = sitk.GetImageFromArray(np.array(self.dose * RBE, dtype=np.single))
        dose_img.SetOrigin(np.asarray(self.origin, dtype=float).tolist())
        dose_img.SetSpacing(np.asarray(self.spacing, dtype=float).tolist())

        if individual_beams:
            for i, beam_dose in enumerate(self.beam_doses):
                dose_img = sitk.GetImageFromArray(np.array(beam_dose * RBE, dtype=np.single))
                dose_img.SetOrigin(np.asarray(self.origin, dtype=float).tolist())
                dose_img.SetSpacing(np.asarray(self.spacing, dtype=float).tolist())
                fw.SetFileName(dose_path.replace(".nrrd", "_beam%02i.nrrd" % (i+1)))
                fw.Execute(dose_img)
        else:
            fw.SetFileName(dose_path)
            fw.Execute(dose_img)

    def writeCTNRRD(self, ct_path):

        if not ct_path.endswith(".nrrd"):
            raise Exception("CT path must have .nrrd extension")

        fw = sitk.ImageFileWriter()
        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(np.asarray(self.origin, dtype=float).tolist())
        HU_img.SetSpacing(np.asarray(self.spacing, dtype=float).tolist())

        fw.SetFileName(ct_path)
        fw.Execute(HU_img)

    def writeCTNIFTI(self, ct_path):
        if not ct_path.endswith(".nii.gz"):
            raise Exception("CT path must have .nii.gz extension")

        HU_img = sitk.GetImageFromArray(self.HU)
        HU_img.SetOrigin(np.asarray(self.origin, dtype=float).tolist())
        HU_img.SetSpacing(np.asarray(self.spacing, dtype=float).tolist())

        fw = sitk.ImageFileWriter()
        fw.SetFileName(ct_path)
        fw.Execute(HU_img)

    def createCubePhantom(self, size=(138, 138, 138), spacing=3.0):
        """Create an air-backed water phantom.

        Args:
            size: NumPy array shape in ``(z, y, x)`` order.
            spacing: Isotropic voxel spacing in millimetres.
        """
        size = tuple(int(value) for value in size)
        if len(size) != 3 or any(value <= 0 for value in size):
            raise ValueError("size must contain three positive values in (z, y, x) order")
        spacing = float(spacing)
        if spacing <= 0.0:
            raise ValueError("spacing must be positive")

        nz, ny, nx = size
        # Retain the historical half-extent convention so the 138-cube
        # reference remains reproducible; only map those extents to xyz here.
        self.origin = np.array(
            [-nx * spacing / 2.0, -ny * spacing / 2.0, -nz * spacing / 2.0]
        )
        self.spacing = np.array([spacing, spacing, spacing])
        self.size = np.array(size, dtype=np.int64)
        edge = round(10.0 / spacing)
        self.HU = np.ones(size, dtype=np.single) * -1000.0
        self.HU[edge:-edge, edge:-edge, edge:-edge] = 0.0


class Beam:

    def __init__(self):
        self.iso = np.array([0.0, 0.0, 0.0], dtype=np.single)
        self.gantry_angle = 0.0
        self.collimator_angle = 0.0
        self.couch_angle = 0.0
        self.BeamName = None
        self.BeamDescription = None


class Plan:

    def __init__(self):
        self.n_beams = 0
        self.n_fractions = 1
        self.beam_list = []
        self.RTPlanLabel = None
        self.Prescriptions = []

    def addPrescription(self, TargetPrescriptionDose, ROIName, TargetUnderdoseVolumeFraction):
        rx = Prescription()
        rx.TargetPrescriptionDose = TargetPrescriptionDose
        rx.ROIName = ROIName
        rx.TargetUnderdoseVolumeFraction = TargetUnderdoseVolumeFraction
        self.Prescriptions.append(rx)

    def addBeam(self, beam):
        if not beam.BeamName:
            beam.BeamName = f'PBS_Beam{self.n_beams + 1}'
        if not beam.BeamDescription:
            beam.BeamDescription = f'PBS_Beam {self.n_beams + 1}'
        self.beam_list.append(beam)
        self.n_beams += 1
