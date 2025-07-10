# /// script
# dependencies = [
#   "napari[all]",
#   "pandas",
#   "magicgui",
#   "ome-zarr",
# ]
# python = "3.12"
# ///
"""
Napari Data Viewer for Stem Cell Project OME-Zarr Data

This viewer provides a GUI for loading and visualizing OME-Zarr microscopy data
with associated labels and tracking information from stem cell experiments.

Features:
- Load OME-Zarr data with multiple channels and labels
- Navigate between different projects and FOVs
- Load cell tracking data with trajectory visualization
- Support for grayscale and categorical label images
"""

# Standard library imports
import os
import pickle
import lzma
import glob
import argparse

# Third-party imports
import napari
import pandas as pd
from magicgui import magicgui
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from qtpy.QtWidgets import QLabel, QSizePolicy

# Configuration constants
UPPER_QUANTILE = 0.95  # Upper quantile for distance clipping
LOWER_QUANTILE = 0.05  # Lower quantile for distance clipping


# --- Parse command-line arguments for base path ---
def get_base_path_and_experiment():
    parser = argparse.ArgumentParser(
        description="Napari Data Viewer for Stem Cell Project OME-Zarr Data"
    )
    parser.add_argument(
        "base_path",
        type=str,
        nargs="?",
        default=".",
        help="Base path to the stem cell project data (default: current directory)",
    )
    parser.add_argument(
        "--data_analysed",
        type=str,
        default=None,
        help="Direct path to Analysed_Data or Analysed_Data2 folder containing FOVs. Overrides base_path if set.",
    )
    args, _ = parser.parse_known_args()
    if args.data_analysed:
        analysed_path = os.path.abspath(args.data_analysed)
        experiment = os.path.basename(os.path.dirname(analysed_path))
        return (
            os.path.dirname(os.path.dirname(analysed_path)),
            experiment,
            analysed_path,
        )
    else:
        base_path = os.path.abspath(args.base_path)
        experiment = os.path.basename(base_path)
        parent_path = os.path.dirname(base_path)
        return parent_path, experiment, None


BASE_PATH_STEM_CELL, DEFAULT_EXPERIMENT, ANALYSED_DATA_PATH = (
    get_base_path_and_experiment()
)


def find_subfolders_with_analysed_data(directory):
    """
    Find all experiment folders that contain FOV folders directly or in Analysed_Data/Analysed_Data2.

    Args:
        directory (str): Path to the directory to search

    Returns:
        list: List of experiment folder paths that contain FOV data
    """
    result = []
    try:
        for subfolder in os.listdir(directory):
            subfolder_path = os.path.join(directory, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            # Check for FOV folders directly in experiment
            has_fov = any(
                os.path.isdir(os.path.join(subfolder_path, f)) and f.startswith("FOV_")
                for f in os.listdir(subfolder_path)
            )
            # Or check for FOV folders in Analysed_Data or Analysed_Data2
            for analysed in ["Analysed_Data", "Analysed_Data2"]:
                analysed_path = os.path.join(subfolder_path, analysed)
                if os.path.isdir(analysed_path):
                    has_fov = has_fov or any(
                        os.path.isdir(os.path.join(analysed_path, f))
                        and f.startswith("FOV_")
                        for f in os.listdir(analysed_path)
                    )
            if has_fov:
                result.append(subfolder_path)
    except (OSError, PermissionError) as e:
        print(f"Error accessing directory {directory}: {e}")
    return result


def sort_key(s):
    """
    Custom sorting key for FOV names.

    Args:
        s (str): FOV name (e.g., "FOV_1", "FOV_10")

    Returns:
        int: Sorting key (-1 for FOV_0, otherwise the numeric part)
    """
    if s == "FOV_0":
        return -1
    try:
        return int(s.split("_")[1])
    except (IndexError, ValueError):
        return float("inf")  # Put invalid names at the end


def find_fov_choices(project_path):
    """
    Find all FOV (Field of View) folders for a given project, searching in:
    - experiment/FOV_*
    - experiment/Analysed_Data/FOV_*
    - experiment/Analysed_Data2/FOV_*
    If ANALYSED_DATA_PATH is set, only search there.
    """
    if ANALYSED_DATA_PATH and os.path.isdir(ANALYSED_DATA_PATH):
        try:
            fovs = [
                f
                for f in os.listdir(ANALYSED_DATA_PATH)
                if os.path.isdir(os.path.join(ANALYSED_DATA_PATH, f))
                and f.startswith("FOV_")
            ]
            return sorted(fovs, key=sort_key)
        except Exception as e:
            print(f"Error accessing FOVs in {ANALYSED_DATA_PATH}: {e}")
            return []
    # ...existing code for normal search...
    base_folder = os.path.join(BASE_PATH_STEM_CELL, project_path)
    fov_dirs = []
    try:
        if os.path.isdir(base_folder):
            fov_dirs.extend(
                f
                for f in os.listdir(base_folder)
                if os.path.isdir(os.path.join(base_folder, f)) and f.startswith("FOV_")
            )
        for analysed in ["Analysed_Data", "Analysed_Data2"]:
            analysed_path = os.path.join(base_folder, analysed)
            if os.path.isdir(analysed_path):
                fov_dirs.extend(
                    f
                    for f in os.listdir(analysed_path)
                    if os.path.isdir(os.path.join(analysed_path, f))
                    and f.startswith("FOV_")
                )
        fovs = sorted(set(fov_dirs), key=sort_key)
        return fovs
    except (OSError, PermissionError) as e:
        print(f"Error accessing FOV directory {base_folder}: {e}")
        return []


def get_fov_folder(project, fov):
    """
    Return the full path to the FOV folder, searching in:
    - experiment/FOV_*
    - experiment/Analysed_Data/FOV_*
    - experiment/Analysed_Data2/FOV_*
    If ANALYSED_DATA_PATH is set, only search there.
    """
    if ANALYSED_DATA_PATH and os.path.isdir(ANALYSED_DATA_PATH):
        direct_path = os.path.join(ANALYSED_DATA_PATH, fov)
        if os.path.isdir(direct_path):
            return direct_path
        return None
    # ...existing code for normal search...
    base_folder = os.path.join(BASE_PATH_STEM_CELL, project)
    direct_path = os.path.join(base_folder, fov)
    if os.path.isdir(direct_path):
        return direct_path
    for analysed in ["Analysed_Data", "Analysed_Data_2"]:
        analysed_path = os.path.join(base_folder, analysed, fov)
        if os.path.isdir(analysed_path):
            return analysed_path
    return None


# Initialize data folders and FOV choices
if ANALYSED_DATA_PATH and os.path.isdir(ANALYSED_DATA_PATH):
    data_folders = [ANALYSED_DATA_PATH]
    # Set the project name to the experiment (parent of Analysed_Data folder)
    data_folder_names = [os.path.basename(os.path.dirname(ANALYSED_DATA_PATH))]
else:
    if DEFAULT_EXPERIMENT and os.path.isdir(
        os.path.join(BASE_PATH_STEM_CELL, DEFAULT_EXPERIMENT)
    ):
        data_folders = [os.path.join(BASE_PATH_STEM_CELL, DEFAULT_EXPERIMENT)]
    else:
        data_folders = find_subfolders_with_analysed_data(BASE_PATH_STEM_CELL)
    data_folder_names = [os.path.basename(folder) for folder in data_folders]


# --- Optionally add extra data folders from a text file ---
def get_extra_folders_file():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--extra-folders-file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "extra_data_folders.txt"),
        help="Path to a text file with additional data folders (default: extra_data_folders.txt in script directory)",
    )
    args, _ = parser.parse_known_args()
    return args.extra_folders_file


EXTRA_DATA_FOLDERS_FILE = get_extra_folders_file()
extra_data_folders = []
if EXTRA_DATA_FOLDERS_FILE and os.path.exists(EXTRA_DATA_FOLDERS_FILE):
    with open(EXTRA_DATA_FOLDERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            folder = line.strip()
            if not folder:
                continue
            # Accept both absolute and relative paths
            folder_path = (
                folder
                if os.path.isabs(folder)
                else os.path.join(BASE_PATH_STEM_CELL, folder)
            )
            if os.path.isdir(folder_path) and folder_path not in data_folders:
                extra_data_folders.append(folder_path)
            elif not os.path.isdir(folder_path):
                print(
                    f"Warning: Extra data folder does not exist or is not a directory: {folder_path}"
                )
# Merge extra folders (if any) with found folders
if extra_data_folders:
    data_folders.extend(extra_data_folders)
    # Remove duplicates
    data_folders = list(dict.fromkeys(data_folders))
# Update data_folder_names after merging
data_folder_names = [os.path.basename(folder) for folder in data_folders]

# Handle case where no data folders are found
if not data_folder_names:
    data_folder_names = ["No data found"]
    print("No data folders found.")
    exit(1)

# Initialize current FOV selection
current_fovs = find_fov_choices(data_folder_names[0]) if data_folder_names else []
current_fov = current_fovs[0] if current_fovs else None
project_c = data_folder_names[0] if data_folder_names else None


@magicgui(
    project={
        "label": "Project: ",
        "choices": data_folder_names,
        "value": data_folder_names[0] if data_folder_names else None,
    },
    fov={
        "label": "Position: ",
        "choices": find_fov_choices(data_folder_names[0]) if data_folder_names else [],
        "value": current_fov,
    },
    next_fov={"widget_type": "PushButton", "label": "Next FOV ->"},
    previous_fov={"widget_type": "PushButton", "label": "Previous FOV <-"},
    load_tracking_data_button={
        "label": "Load Tracking Data",
        "widget_type": "PushButton",
        "visible": False,
    },
    call_button="Load Data",
)
def load_data_widget(
    project,
    fov,
    next_fov: bool = False,
    previous_fov: bool = False,
    load_tracking_data_button: bool = False,
):
    """
    Main widget function for loading OME-Zarr data and labels into Napari.

    Args:
        project (str): Selected project name
        fov (str): Selected FOV name
        next_fov (bool): Flag for next FOV button (unused in function body)
        previous_fov (bool): Flag for previous FOV button (unused in function body)
        load_tracking_data_button (bool): Flag for loading tracking data button (unused in function body)
    """
    global current_fov, project_c

    # Update global state
    current_fov = fov
    project_c = project

    # Clear existing layers
    viewer.layers.clear()

    # Find the FOV folder (experiment/FOV_* or experiment/Analysed_Data/FOV_* or experiment/Analysed_Data2/FOV_*)
    url = get_fov_folder(project, fov)
    if url is None:
        print(f"Error: Could not find FOV folder for {project} {fov}")
        return

    # Check if the path exists before trying to load
    if not os.path.exists(url):
        print(f"Error: Path does not exist: {url}")
        return

    if not os.path.isdir(url):
        print(f"Error: Path is not a directory: {url}")
        return

    try:
        # Load OME-Zarr data
        print(f"Attempting to load OME-Zarr data from: {url}")

        # Check if this looks like a valid OME-Zarr directory
        zarr_files = [f for f in os.listdir(url) if f.startswith(".z")]
        print(f"Zarr files found: {zarr_files}")

        if not any(f in zarr_files for f in [".zattrs", ".zgroup"]):
            print(f"Warning: No .zattrs or .zgroup files found in {url}")
            print(
                f"Directory contents: {os.listdir(url)[:10]}..."
            )  # Show first 10 items

        parsed_url = parse_url(url)
        if parsed_url is None:
            print(f"Error: parse_url returned None for {url}")
            print("This directory does not appear to contain valid OME-Zarr data.")
            return

        reader = Reader(parsed_url)
        nodes = list(reader())

        if not nodes:
            print(f"No data found in {url}")
            return

        # Add image data (first node contains raw images)
        channel_axis = next(
            (
                i
                for i, item in enumerate(nodes[0].metadata["axes"])
                if item["name"] == "c"
            ),
            None,
        )

        viewer.add_image(
            nodes[0].data,
            channel_axis=channel_axis,
            name=(
                nodes[0].metadata["channel_names"]
                if "channel_names" in nodes[0].metadata
                else (
                    nodes[0].metadata["channel"]
                    if "channel" in nodes[0].metadata
                    else None
                )
            ),
            contrast_limits=nodes[0].metadata["contrast_limits"],
            visible=False,
        )

        # Add label data (subsequent nodes contain labels)
        print("Loading labels...")
        labels = nodes[1].zarr.root_attrs["labels"]

        for i in range(2, len(nodes)):
            try:
                # Check if this is a grayscale label or categorical label
                is_grayscale = (
                    nodes[i]
                    .zarr.root_attrs["multiscales"][0]["metadata"]
                    .get("is_grayscale_label", False)
                )
                label_name = labels[i - 2]

                if is_grayscale:
                    viewer.add_image(nodes[i].data, name=label_name, visible=False)
                else:
                    viewer.add_labels(nodes[i].data, name=label_name, visible=False)
            except Exception as e:
                print(f"Error loading label {i-2}: {e}")

        # Check if tracking data is available
        graph_pattern = os.path.join(BASE_PATH_STEM_CELL, project, f"{fov}_graph_*.xz")
        has_graph = bool(glob.glob(graph_pattern))
        load_data_widget.load_tracking_data_button.visible = has_graph

        # Reset viewer to first timepoint
        viewer.dims.set_current_step(0, 0)

    except Exception as e:
        print(f"Error loading data from {url}: {e}")


@load_data_widget.load_tracking_data_button.changed.connect
def load_tracking_data(value):
    """
    Load and display cell tracking data as tracks in Napari.

    Args:
        value: Widget value (unused)
    """
    graph_file_path = os.path.join(
        BASE_PATH_STEM_CELL, project_c, f"{current_fov}_graph_2.xz"
    )

    if not os.path.exists(graph_file_path):
        print(f"Graph file not found: {graph_file_path}")
        return

    try:
        # Load tracking graph
        with lzma.open(graph_file_path, "rb") as f:
            graph = pickle.load(f)

        # Load tracking DataFrame
        tracks_df = pd.read_pickle(
            os.path.join(
                BASE_PATH_STEM_CELL,
                project_c,
                f"{current_fov}_df_tracks_2.xz",
            ),
            compression="xz",
        )

        # Clip distance values to remove outliers
        if "distance" in tracks_df.columns:
            upper_q = tracks_df["distance"].quantile(UPPER_QUANTILE)
            lower_q = tracks_df["distance"].quantile(LOWER_QUANTILE)
            tracks_df["distance_clip"] = tracks_df["distance"].clip(
                lower=lower_q, upper=upper_q
            )
            distance = tracks_df["distance_clip"].values
            features = {"distance": distance}
        else:
            features = {}

        # Add tracks to viewer
        viewer.add_tracks(
            tracks_df[["track_id", "t", "y", "x"]].values,
            graph=graph,
            features=features,
        )

    except Exception as e:
        print(f"Error loading tracking data: {e}")


@load_data_widget.next_fov.changed.connect
def set_next(_):
    """
    Navigate to the next FOV in the list.

    Args:
        _: Widget value (unused)
    """
    fov_choices = find_fov_choices(project_c)

    if current_fov in fov_choices:
        current_index = fov_choices.index(current_fov)
        if current_index < len(fov_choices) - 1:
            load_data_widget.fov.value = fov_choices[current_index + 1]
    elif current_fov is None and fov_choices:
        load_data_widget.fov.value = fov_choices[0]

    # Trigger data loading
    load_data_widget.call_button.clicked.emit()


@load_data_widget.previous_fov.changed.connect
def set_previous(_):
    """
    Navigate to the previous FOV in the list.

    Args:
        _: Widget value (unused)
    """
    fov_choices = find_fov_choices(project_c)

    if current_fov in fov_choices:
        current_index = fov_choices.index(current_fov)
        if current_index > 0:
            load_data_widget.fov.value = fov_choices[current_index - 1]
    elif current_fov is None and fov_choices:
        load_data_widget.fov.value = fov_choices[0]

    # Trigger data loading
    load_data_widget.call_button.clicked.emit()


@load_data_widget.project.changed.connect
def on_project_change(_=None):
    """
    Update FOV choices when project selection changes.

    Args:
        _: Event parameter (unused)
    """
    fovs = find_fov_choices(load_data_widget.project.value)
    load_data_widget.fov.choices = fovs
    # Note: Accessing protected member is not ideal, but required for magicgui
    load_data_widget.fov._default_choices = fovs  # noqa: SLF001
    load_data_widget.fov.value = fovs[0] if fovs else None


# Initialize Napari viewer and GUI
viewer = napari.viewer.Viewer()

# Set up the GUI widget
load_fov_widget = load_data_widget
load_data_widget.max_width = 500

# Add widget to viewer as a docked widget
load_data_block = viewer.window.add_dock_widget(load_fov_widget, name="Load Data")

# Main execution
if __name__ == "__main__":
    napari.run()
