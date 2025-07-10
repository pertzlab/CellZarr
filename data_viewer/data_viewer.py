# /// script
# dependencies = [
#   "napari",
#   "pandas",
#   "magicgui",
#   "ome-zarr",
#   "qtpy",
#   "PyQt5",
# ]
# python = "3.11"
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

# Configuration constants
UPPER_QUANTILE = 0.95  # Upper quantile for distance clipping
LOWER_QUANTILE = 0.05  # Lower quantile for distance clipping


# --- Parse command-line arguments for base path ---
def get_base_path():
    parser = argparse.ArgumentParser(
        description="Napari Data Viewer for Stem Cell Project OME-Zarr Data"
    )
    parser.add_argument(
        "base_path",
        nargs="?",
        type=str,
        default=".",
        help="Base path to the stem cell project data or specific experiment folder",
    )
    args, _ = parser.parse_known_args()
    return os.path.abspath(args.base_path)


BASE_PATH_STEM_CELL = get_base_path()


def find_subfolders_with_analysed_data(directory):
    """
    Find all subfolders in the given directory that contain analyzed data.
    Supports different folder structures:
    1. topfolder/experiment/FOV_* (direct FOV folders)
    2. topfolder/experiment/Analysed_Data*/FOV_* (FOV folders in Analysed_Data subfolder)

    Args:
        directory (str): Path to the directory to search

    Returns:
        dict: Dictionary with experiment names as keys and their base paths as values
    """
    result = {}

    # If the directory itself contains FOV folders, treat it as a single experiment
    if contains_fov_folders(directory):
        experiment_name = os.path.basename(directory)
        result[experiment_name] = directory
        return result

    # Check if the directory contains only Analysed_Data* folders with FOV data
    # This handles the case when pointing directly to an experiment folder
    analysed_data_folders = []
    other_folders_with_fovs = []

    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if not os.path.isdir(item_path):
                continue

            if item.startswith("Analysed_Data") and contains_fov_folders(item_path):
                analysed_data_folders.append((item, item_path))
            elif contains_fov_folders(item_path) and not item.startswith("tracking"):
                other_folders_with_fovs.append((item, item_path))

        # If we found Analysed_Data folders, use only those
        if analysed_data_folders:
            experiment_name = os.path.basename(directory)
            # Use the first Analysed_Data folder found
            result[experiment_name] = analysed_data_folders[0][1]
            return result

        # If no Analysed_Data folders but other folders with FOVs, treat as experiments
        if other_folders_with_fovs:
            for folder_name, folder_path in other_folders_with_fovs:
                result[folder_name] = folder_path
            return result

    except (OSError, PermissionError) as e:
        print(f"Error accessing directory {directory}: {e}")
        return result

    # If we reach here, treat subfolders as separate experiments (topfolder scenario)
    try:
        for subfolder in os.listdir(directory):
            subfolder_path = os.path.join(directory, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Check if this experiment folder directly contains FOV folders
            if contains_fov_folders(subfolder_path):
                result[subfolder] = subfolder_path
            else:
                # Only check for Analysed_Data* subfolders, ignore all other folders (including tracking)
                for item in os.listdir(subfolder_path):
                    item_path = os.path.join(subfolder_path, item)
                    if (
                        os.path.isdir(item_path)
                        and item.startswith("Analysed_Data")
                        and contains_fov_folders(item_path)
                    ):
                        result[subfolder] = item_path
                        break

    except (OSError, PermissionError) as e:
        print(f"Error accessing directory {directory}: {e}")
    return result


def contains_fov_folders(directory):
    """
    Check if a directory contains FOV folders.

    Args:
        directory (str): Path to check

    Returns:
        bool: True if directory contains FOV folders
    """
    try:
        for item in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, item)) and item.startswith("FOV_"):
                return True
    except (OSError, PermissionError):
        pass
    return False


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


def find_fov_choices(experiment_name):
    """
    Find all FOV (Field of View) folders for a given experiment.

    Args:
        experiment_name (str): Name of the experiment

    Returns:
        list: Sorted list of FOV names
    """
    # Get the base path for this experiment
    base_folder = experiment_paths.get(experiment_name)
    if not base_folder:
        return []

    try:
        fovs = [
            os.path.basename(f)
            for f in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, f)) and f.startswith("FOV_")
        ]
        fovs = sorted(fovs, key=sort_key)
        return fovs
    except (OSError, PermissionError) as e:
        print(f"Error accessing FOV directory {base_folder}: {e}")
        return []


# Initialize data folders and FOV choices
experiment_data = find_subfolders_with_analysed_data(BASE_PATH_STEM_CELL)
experiment_names = list(experiment_data.keys())
experiment_paths = experiment_data


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
extra_data_folders = {}
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
            if os.path.isdir(folder_path):
                folder_name = os.path.basename(folder_path)
                if folder_name not in experiment_paths:
                    extra_data_folders[folder_name] = folder_path
            elif not os.path.isdir(folder_path):
                print(
                    f"Warning: Extra data folder does not exist or is not a directory: {folder_path}"
                )

# Merge extra folders (if any) with found folders
if extra_data_folders:
    experiment_paths.update(extra_data_folders)
    experiment_names = list(experiment_paths.keys())

# Handle case where no data folders are found
if not experiment_names:
    experiment_names = ["No data found"]
    print("No data folders found.")
    exit(1)

print(f"Found experiments: {experiment_names}")
print(f"Base path: {BASE_PATH_STEM_CELL}")
for exp_name, exp_path in experiment_paths.items():
    fov_count = len(find_fov_choices(exp_name))
    print(f"  {exp_name}: {exp_path} ({fov_count} FOVs)")

# Initialize current FOV selection
current_fovs = find_fov_choices(experiment_names[0]) if experiment_names else []
current_fov = current_fovs[0] if current_fovs else None
project_c = experiment_names[0] if experiment_names else None


@magicgui(
    project={
        "label": "Project: ",
        "choices": experiment_names,
        "value": experiment_names[0] if experiment_names else None,
    },
    fov={
        "label": "Position: ",
        "choices": find_fov_choices(experiment_names[0]) if experiment_names else [],
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

    # Construct path to OME-Zarr data
    experiment_path = experiment_paths.get(project)
    if not experiment_path:
        print(f"Error: No path found for experiment: {project}")
        return

    url = os.path.join(experiment_path, fov)

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
            name=nodes[0].metadata["channel_names"],
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
        experiment_path = experiment_paths.get(project)
        graph_pattern = os.path.join(experiment_path, f"{fov}_graph_*.xz")
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
    experiment_path = experiment_paths.get(project_c)
    if not experiment_path:
        print(f"Error: No path found for experiment: {project_c}")
        return

    graph_file_path = os.path.join(experiment_path, f"{current_fov}_graph_2.xz")

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
                experiment_path,
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
