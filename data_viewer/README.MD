# Napari Data Viewer for Stem Cell Project OME-Zarr Data

This is a specialized Napari-based viewer for visualizing OME-Zarr microscopy data from data generated through the CellZarr pipeline. The viewer provides an intuitive GUI for loading and analyzing multi-channel time-lapse data with associated labels and cell tracking information.

## Features

- **Multi-channel OME-Zarr support**: Load and visualize time-lapse microscopy data with multiple channels
- **Label visualization**: Support for both categorical labels (segmentation masks) and grayscale label images (e.g., ERK activity overlays)
- **Cell tracking**: Load and visualize cell trajectories with distance-based coloring
- **Interactive navigation**: Easy switching between different experiments and Fields of View (FOVs)
- **Automatic data discovery**: Automatically finds and lists available experiments and FOVs

## Data Structure

The viewer supports flexible data organization and can automatically detect different folder structures:

### Structure 1: Direct FOV folders in experiment
```
experiment_folder/
├── FOV_0/          # OME-Zarr data for field of view 0
├── FOV_1/          # OME-Zarr data for field of view 1
├── FOV_2/
├── FOV_1_graph_*.xz    # Optional tracking data for FOV 1
├── FOV_1_df_tracks_*.xz # Optional tracking data for FOV 1
└── ...
```

### Structure 2: FOV folders in Analysed_Data subfolder
```
experiment_folder/
├── Analysed_Data/    # or Analysed_Data_1, etc.
│   ├── FOV_0/          # OME-Zarr data for field of view 0
│   ├── FOV_1/          # OME-Zarr data for field of view 1
│   ├── FOV_1_graph_*.xz    # Optional tracking data for FOV 1
│   ├── FOV_1_df_tracks_*.xz # Optional tracking data for FOV 1
│   └── ...
├── tracking/           # This folder is ignored
└── other_folders/      # Other folders are ignored
```

### Structure 3: Multiple experiments in top folder
```
data_directory/
├── experiment1/
│   ├── FOV_0/          # OME-Zarr data for field of view 0
│   ├── FOV_1/          # OME-Zarr data for field of view 1
│   └── ...
├── experiment2/
│   ├── Analysed_Data/
│   │   ├── FOV_0/
│   │   ├── FOV_1/
│   │   └── ...
│   └── tracking/       # This folder is ignored
└── ...
```

Each FOV folder should contain:
- OME-Zarr image data with multiple channels and labels
- Optional tracking data (`FOV_X_graph_*.xz` and `FOV_X_df_tracks_*.xz` files)

## Installation and Usage

### Using uv (Recommended)

The viewer uses inline dependency specification, so all required packages are automatically installed when using uv:

```bash
uv run data_viewer.py
```

### Manual Installation

If you prefer to install dependencies manually, use the requirements.txt file:

```bash
pip install -r requirements.txt
python data_viewer.py
```

## Command Line Options

The viewer supports flexible command-line usage for different scenarios:

### Basic Usage
```bash
# Run from current directory (searches for experiments in current folder)
uv run data_viewer.py

# Point to a folder containing multiple experiments
uv run data_viewer.py PATH_TO_EXPERIMENT_FOLDERS

# Point directly to a specific experiment folder
uv run data_viewer.py EXPERIMENT_FOLDER_WITH_FOVS
```

### Examples
```bash
# Load all experiments from a data directory
uv run data_viewer.py /path/to/data/StemCellProject

# Load a specific experiment directly
uv run data_viewer.py /path/to/data/StemCellProject/20220517_starve21h_FGF20_200

# Using Windows network paths
uv run data_viewer.py "\\server\share\StemCellProject\experiment1"
```

### Advanced Options
```bash
# Use custom extra folders file
uv run data_viewer.py --extra-folders-file /path/to/custom_folders.txt

# Get help
uv run data_viewer.py --help
```

### Extra Data Folders

By default, the viewer looks for an `extra_data_folders.txt` file in the same directory as the script. This file can contain additional data folder paths (one per line) to include experiments from different locations:

```
/path/to/additional/experiment1
/path/to/another/location/experiment2
../relative/path/to/experiment3
```

### Data Organization Notes

- **Analysed_Data Priority**: When pointing to an experiment folder, only `Analysed_Data*` subfolders are considered, other folders (like `tracking`) are ignored
- **Automatic Detection**: The viewer automatically detects the folder structure and adapts accordingly
- **FOV Sorting**: FOV folders are sorted numerically (FOV_0, FOV_1, FOV_2, ..., FOV_10, etc.)

## GUI Interface

The viewer provides an intuitive interface with the following controls:

- **Project dropdown**: Select from automatically discovered experiments
- **Position dropdown**: Choose FOV within the selected experiment
- **Navigation buttons**:
  - "Previous FOV <-": Navigate to previous field of view
  - "Next FOV ->": Navigate to next field of view
- **Load Data button**: Load selected experiment and FOV
- **Load Tracking Data button**: Appears when tracking data is available for the current FOV

## Data Loading Process

1. **Image Data**: Multi-channel time-lapse data is loaded with automatic contrast adjustment
2. **Label Data**: Both categorical labels (segmentation) and grayscale labels (overlays) are supported
3. **Tracking Data**: When available, cell trajectories are loaded with distance-based visualization

## Technical Details

- **OME-Zarr Extension**: Supports grayscale label images through custom metadata (`is_grayscale_label` flag)
- **Data Extraction**: Uses the standard `ome-zarr` library for data access
- **Tracking Format**: Supports compressed pickle files (`.xz`) for trajectory data
- **Performance**: Optimized for large time-lapse datasets with efficient memory usage

## Dependencies

- napari
- pandas
- magicgui
- ome-zarr
- qtpy
- PySide2
