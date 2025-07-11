# CellZarr: High-Content Image Analysis Pipeline for Live Imaging Timelapse Experiments

CellZarr is a comprehensive and reproducible workflow for high-content image analysis of live cell imaging timelapse experiments. The pipeline processes microscopy data from raw ND2 files to quantitative feature extraction, enabling downstream biological analysis using modern, scalable tools and formats.

## Pipeline Overview

The workflow is designed to be highly modular and consists of the following main steps:

1. **ND2 to OME-Zarr conversion**: Convert raw ND2 microscopy files to the OME-Zarr format for scalable, cloud-ready storage and analysis.
2. **Colony segmentation using ConvPaint**: Identify and segment stem cell colonies in the images using a deep learning-based approach.
3. **Nucleus segmentation using StarDist**: Detect and segment individual nuclei within colonies for single-cell analysis.
4. **Cell Tracking**: Track individual cells over time to study dynamic behaviors using ultrack.
5. **Feature Extraction**: Quantify spatial features and extract relevant biological markers (e.g., ERK, Oct4) for each cell.

The workflow is highly modular, making it straightforward to adapt to different datasets or analysis needs. Once the ND2 files have been converted to OME-Zarr, the subsequent steps can be performed independently, allowing you to skip or repeat steps as required for your analysis.

## Key Features

- **Scalable Processing**: Extensive use of Dask for parallel and distributed computing
- **Modern Data Format**: OME-Zarr for efficient storage and cloud compatibility
- **Interactive Visualization**: Custom Napari-based data viewer for exploring results
- **Configurable**: Easy configuration through settings files

## Installation and Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager

**Note**: Thanks to using uv as a package manager, you don't need to install Python or Conda separately. uv automatically manages Python installations and virtual environments for you.

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pertzlab/CellZarr
   cd CellZarr
   ```

2. **Install all dependencies**:
   ```bash
   uv sync --all-extras
   ```

   This command automatically creates a virtual environment (`.venv`) in your project directory and installs all required dependencies. This virtual environment can be used directly in VS Code for running the Jupyter notebooks.

3. **PyQt5 fix** (required due to PyQt5 compatibility issues):
   ```bash
   uv pip install pyqt5
   ```

### Selective Installation

If you don't need all pipeline components, you can install only specific extras:

#### For StarDist nucleus segmentation only:
```bash
uv sync --extra stardist_seg
```

#### For ConvPaint colony segmentation only:
```bash
uv sync --extra colony_seg
uv pip install pyqt5  # Required on Windows
```

#### Base installation (without segmentation models):
```bash
uv sync
```

**Virtual Environment Integration**: All `uv sync` commands automatically create and manage a virtual environment (`.venv`) in your project directory. VS Code will automatically detect this environment and can use it for running Jupyter notebooks and Python scripts.

## Configuration

Configuration options such as output paths and scaling parameters can be easily adjusted in:
- `configuration/settings.py` - General pipeline settings
- `configuration/dask.py` - Dask cluster configuration

## Pipeline Components

### 1. ND2 to OME-Zarr Conversion (`01_ND2_to_OME-ZARR.ipynb`)

The first step converts raw ND2 microscopy files into the OME-Zarr format. ND2 is a proprietary file format commonly used for storing high-content microscopy data. OME-Zarr is an open, scalable, and cloud-compatible format that enables efficient storage, access, and analysis of large multidimensional image datasets.

In this step, the ND2 file is loaded, relevant metadata is extracted, and each field of view (FOV) is saved as a separate OME-Zarr dataset.

### 2. Colony Segmentation (`02_Colony_Segmentation.ipynb`)

Uses ConvPaint, a deep learning-based approach, to identify and segment stem cell colonies in the images. This step requires the `colony_seg` extra dependencies.

### 3. Nucleus Segmentation (`03_Nucleus_Segmentation.ipynb`)

Employs StarDist to detect and segment individual nuclei within colonies for single-cell analysis. This enables downstream single-cell feature extraction and tracking.

### 4. Cell Tracking

Cell tracking is implemented using ultrack and can be executed in two ways:

#### Python Wrapper (`04_Cell_Tracking_ultrack.py` and `04_Cell_Tracking_ultrack_all_fovs.py`)
- Direct execution through Python scripts
- Suitable for local processing or smaller datasets

#### SLURM Batch Processing (`04_Cell_Tracking_slurm.sh`)
- For high-performance computing environments
- Recommended for large datasets with many FOVs
- Requires SLURM workload manager

**Gurobi License Recommendation**: For optimal performance, we recommend obtaining a Gurobi license:
- Free for academic use at [gurobi.com](https://www.gurobi.com)
- Best compatibility with WSL license: [Web License Service](https://www.gurobi.com/features/web-license-service/)

### 5. Feature Extraction (`05_Feature_Extraction_ERK_Oct4.ipynb`)

Quantifies spatial features and extracts relevant biological markers (e.g., ERK, Oct4) for each cell, enabling downstream biological analysis.

## Data Visualization

The `data_viewer/` folder contains a specialized Napari-based viewer for visualizing the generated OME-Zarr data. This viewer supports:

- Multi-channel OME-Zarr visualization
- Custom grayscale metadata tags for enhanced display
- Label and tracking data overlay
- Interactive navigation between experiments and FOVs

**Important Note on Custom OME-Zarr Labels**: The pipeline generates OME-Zarr files with some labels stored as grayscale images (e.g., biosensor expression overlays) rather than standard categorical labels. While these files can be loaded in Python without issues, the standard OME-Zarr Napari plugin may not properly display these custom grayscale labels (these labels have an additional metadata tag called greyscale set to Tr). Therefore, we strongly recommend using the integrated data viewer provided in this repository, which is specifically designed to handle these custom label types correctly.

### Running the Data Viewer

```bash
cd data_viewer
uv run data_viewer.py [path_to_data]
```

See `data_viewer/README.MD` for detailed usage instructions.

## Parallel Processing with Dask

The pipeline extensively uses Dask to parallelize operations for maximum performance:

- **Distributed Computing**: Automatic scaling across available CPU cores
- **Memory Management**: Efficient handling of large datasets
- **Progress Monitoring**: Real-time progress tracking for long-running operations
- **Configurable**: Cluster settings can be adjusted in `configuration/dask.py`

## File Structure

```
CellZarr/
├── 01_ND2_to_OME-ZARR.ipynb          # ND2 to OME-Zarr conversion
├── 02_Colony_Segmentation.ipynb       # Colony segmentation with ConvPaint
├── 03_Nucleus_Segmentation.ipynb      # Nucleus segmentation with StarDist
├── 04_Cell_Tracking_ultrack.py        # Cell tracking (single FOV)
├── 04_Cell_Tracking_ultrack_all_fovs.py # Cell tracking (all FOVs)
├── 04_Cell_Tracking_slurm.sh          # SLURM batch job for tracking
├── 05_Feature_Extraction_ERK_Oct4.ipynb # Feature extraction
├── pyproject.toml                      # Project dependencies
├── uv.lock                            # Dependency lock file
├── configuration/                      # Configuration files
│   ├── settings.py                    # General settings
│   └── dask.py                        # Dask cluster configuration
├── data_viewer/                       # Napari-based data viewer
│   ├── data_viewer.py                 # Main viewer application
│   └── README.MD                      # Viewer documentation
├── helper_functions/                  # Utility functions
└── models/                           # Pre-trained segmentation models
```

## Usage Examples

### Running Individual Pipeline Steps

1. **Start with ND2 conversion**:
   ```bash
   uv run jupyter lab 01_ND2_to_OME-ZARR.ipynb
   ```

2. **Perform segmentation**:
For cell colony segmentation:
   ```bash
   uv run jupyter lab 02_Colony_Segmentation.ipynb
   ```

For nucleus segmentation:
   ```bash
   uv run jupyter lab 03_Nucleus_Segmentation.ipynb
   ```

3. **Track cells**:
   ```bash
   # For single FOV
   python 04_Cell_Tracking_ultrack.py

   # For all FOVs
   python 04_Cell_Tracking_ultrack_all_fovs.py

   # Or submit SLURM job
   sbatch 04_Cell_Tracking_slurm.sh
   ```

4. **Extract features**:
   ```bash
   jupyter lab 05_Feature_Extraction_ERK_Oct4.ipynb
   ```

### Viewing Results

```bash
cd data_viewer
uv run data_viewer.py /path/to/your/processed/data
```
