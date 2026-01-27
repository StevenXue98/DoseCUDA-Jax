from connect import *
import os

beam_set = get_current("BeamSet")
case = get_current("Case")
patient = get_current("Patient")

energies = [
    70.2,
    71.1,
    72.1,
    73.0,
    73.9,
    74.8,
    75.7,
    76.6,
    77.5,
    78.4,
    79.2,
    80.1,
    81.0,
    81.8,
    82.7,
    83.5,
    84.3,
    85.2,
    86.0,
    86.8,
    87.6,
    88.4,
    89.2,
    90.0,
    90.8,
    91.6,
    92.3,
    93.1,
    93.9,
    94.6,
    95.8,
    96.9,
    98.4,
    99.8,
    101.3,
    102.7,
    104.2,
    105.6,
    107.0,
    108.3,
    109.7,
    111.1,
    112.4,
    113.7,
    115.1,
    116.4,
    117.7,
    119.0,
    120.2,
    121.5,
    123.1,
    124.7,
    126.5,
    128.3,
    130.2,
    132.0,
    133.7,
    135.5,
    137.3,
    139.0,
    140.7,
    142.4,
    144.1,
    146.1,
    148.0,
    150.2,
    152.4,
    154.5,
    156.6,
    158.7,
    160.8,
    163.2,
    165.5,
    168.0,
    170.5,
    173.0,
    175.5,
    177.9,
    180.3,
    182.7,
    185.1,
    187.5,
    189.8,
    192.4,
    194.9,
    197.6,
    200.3,
    203.0,
    205.7,
    208.3,
    210.9,
    213.5,
    216.1,
    218.7,
    221.2,
    223.7,
    226.2,
    228.7]

parent_directory = "<path_to_export_directory>"

for energy in energies:

    set_progress("Computing energy %.1f" % energy)
    beam_set.Beams['iso_n35'].Segments[0].EditNominalEnergy(NominalEnergy=energy)
    beam_set.Beams['iso_n30'].Segments[0].EditNominalEnergy(NominalEnergy=energy)
    beam_set.Beams['iso_n20'].Segments[0].EditNominalEnergy(NominalEnergy=energy)
    beam_set.Beams['iso_n10'].Segments[0].EditNominalEnergy(NominalEnergy=energy)
    beam_set.Beams['iso_0'].Segments[0].EditNominalEnergy(NominalEnergy=energy)

    beam_set.ComputeDose(ComputeBeamDoses=True, DoseAlgorithm="IonMonteCarlo", ForceRecompute=False, RunEntryValidation=True)

    patient.Save()

    set_progress("Exporting energy %.1f" % energy)
    folder_path = os.path.join(parent_directory, "%.1f" % energy)
    try:
        os.mkdir(folder_path)
    except:
        pass
    case.ScriptableDicomExport(ExportFolderPath = folder_path, BeamSets=["SmallGrid:SmallGrid"], PhysicalBeamDosesForBeamSets=["SmallGrid:SmallGrid"], IgnorePreConditionWarnings=True)
