import os
import glob
import natsort


def get_project_path():
    """
    Returns the project path based on the operating system.
    """
    if os.name == "nt":
        return "\\\\izbkingston.izb.unibe.ch\\imaging.data\\PertzLab\\StemCellProject\\"
    else:
        return "/mnt/imaging.data/PertzLab/StemCellProject"


def get_output_path():
    """
    Returns the output path for the analysed data.
    """
    project_path = get_project_path()
    output_path_parts = [
        "20220517_starve21h_FGF20_200",
        "Analysed_Data_2",
    ]
    return os.path.join(project_path, *output_path_parts)


def get_fovs(output_path: str = get_output_path()):
    """
    Returns a sorted list of FOV names from the output path.
    """
    fovs = glob.glob(os.path.join(output_path, "FOV_*"))
    fovsname = []
    for fov in fovs:
        if os.path.isdir(fov):
            fovsname.append(os.path.basename(fov))
    return natsort.natsorted(fovsname)


axis_norm = (0, 1)
channel_h2b = 0
channel_oct4 = 1
channel_erk = 2
SCALING_FACTOR = 1
