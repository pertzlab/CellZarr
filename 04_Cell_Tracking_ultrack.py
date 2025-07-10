# /// script
# requires-python = "<=3.11"
# dependencies = [
#   "ultrack",
#   "ome-zarr",
#   "zarr",
#   "higra",
#   "gurobipy",
#   "natsort",
# ]
# ///
"""
Cell Tracking Script using Ultrack
==================================

This script performs automated cell tracking on microscopy data using the Ultrack library.

USAGE
-----
1. **Command Line:**
   Run the script for a specific field of view (FOV) by providing the FOV index as an argument:
       python 04_Cell_Tracking_ultrack.py 7
   This will process FOV 7. Replace `7` with the desired FOV index.

2. **With SLURM (Cluster):**
   Use the provided SLURM script (04_Cell_Tracking_slurm.sh) to run tracking in parallel for multiple FOVs:
       sbatch 04_Cell_Tracking_slurm.sh
   The SLURM script will submit jobs for FOVs 1 to 48 (adjust as needed), each running this script with the corresponding FOV index.

WHAT THE SCRIPT DOES
--------------------
- Loads segmentation masks for the specified FOV from a Zarr store.
- Checks if tracking has already been performed for this FOV.
- Configures and runs the Ultrack tracker on the masks.
- Exports and saves the tracking results (tracks and graph) as compressed files.
- Writes the tracking labels back into the ome-zarr store.
"""

from ultrack import MainConfig, Tracker
import ultrack

import ome_zarr.reader as ozr
import ome_zarr.io as ozi
import zarr
import os
import pickle
import lzma
import ome_zarr.writer as ozw
from configuration.settings import get_output_path

tracking_version = 0


def main(fov_i: int):
    print(f"Processing FOV {fov_i}")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    fov = f"FOV_{fov_i}"

    dest = os.path.join(get_output_path(), fov)
    store = ozi.parse_url(dest, mode="a").store
    root = zarr.group(store=store)
    nodes = list(ozr.Reader(ozi.parse_url(dest, mode="r"))())

    labels_list = [node.metadata.get("name") for node in nodes[2:]]
    if "tracked" in labels_list:
        print(f"FOV {fov_i} already tracked")
        return
    if "labels" in labels_list:
        i_nucleus_label = labels_list.index("labels") + 2
    elif "nucleus" in labels_list:
        i_nucleus_label = labels_list.index("nucleus") + 2
    else:
        raise ValueError(
            f"FOV {fov_i} does not contain a 'labels' or 'nucleus' label. Available labels: {labels_list}"
        )
    masks = nodes[i_nucleus_label].data[0].compute()
    Y_dim = masks.shape[1]
    X_dim = masks.shape[2]

    parent_output_path = os.path.dirname(get_output_path())
    tracking_folder = os.path.join(parent_output_path, "Tracking", fov)
    if not os.path.exists(tracking_folder):
        os.makedirs(tracking_folder)
    os.chdir(tracking_folder)

    config = MainConfig()
    config.linking_config.max_distance = 55
    config.linking_config.n_workers = 48
    config.tracking_config.n_threads = 48
    config.data_config.n_workers = 48
    config.segmentation_config.n_workers = 48
    config.tracking_config.disappear_weight = -0.02
    config.tracking_config.division_weight = -0.0005
    config.segmentation_config.min_frontier = 0.5

    tracker = Tracker(config=config)
    tracker.track(labels=masks)

    tracks_df, graph = ultrack.core.export.to_tracks_layer(config)
    labels = ultrack.core.export.tracks_to_zarr(config, tracks_df)

    tracks_df.to_pickle(
        os.path.join(get_output_path(), f"{fov}_df_tracks_{tracking_version}.xz"),
        compression="xz",
    )
    with lzma.open(
        os.path.join(get_output_path(), f"{fov}_graph_{tracking_version}.xz"),
        "wb",
    ) as f:
        pickle.dump(graph, f)

    label_name = "tracked"
    if "labels" in root:
        if label_name in root.labels.attrs["labels"]:
            del root["labels"][label_name]
            current_labels = root.labels.attrs["labels"]
            new_labels = [lbl for lbl in current_labels if lbl != label_name]
            root.labels.attrs["labels"] = new_labels
        try:
            del root["labels"][label_name]
        except:
            pass

    ozw.write_labels(
        labels,
        root,
        name=label_name,
        axes="tyx",
        scaler=None,
        chunks=(1, Y_dim, X_dim),
        storage_options={
            "compressor": zarr.storage.Blosc(cname="zstd", clevel=5),
        },
        metadata={"is_grayscale_label": False},
    )


if __name__ == "__main__":
    fov_i_arg = int(os.sys.argv[1])
    main(fov_i_arg)
