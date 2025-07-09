import ome_zarr.reader as ozr  # For reading OME-Zarr data
import ome_zarr.io as ozi  # For OME-Zarr I/O operations
import ome_zarr.writer as ozw  # For writing label data to OME-Zarr
import zarr  # For Zarr storage
import os  # For file path operations
import ome_zarr.scale  # For scaling operations in OME-Zarr

# Utility function to list all labels in an OME-Zarr group
def list_labels(root):
    """
    Lists all labels in the OME-Zarr group.
    
    Parameters:
    root (zarr.Group): The OME-Zarr group to inspect.
    
    Returns:
    list: A list of label names.
    """
    if "labels" in root and "labels" in root.labels.attrs:
        return root.labels.attrs["labels"]
    else:
        return []

def get_label_array(root, label_name):
    """
    Retrieves a label array from the OME-Zarr group.
    
    Parameters:
    root (zarr.Group): The OME-Zarr group containing the label.
    label_name (str): The name of the label to retrieve.
    
    Returns:
    dask.array.Array: The label array if it exists, otherwise None.
    """
    if "labels" in root and label_name in root.labels:
        return root.labels[label_name]
    else:
        assert False, f"Label '{label_name}' does not exist in the OME-Zarr group."

# Utility function to delete a label group from OME-Zarr
def delete_label(root, label_name):
    """
    Deletes a label from the OME-Zarr group.
    
    Parameters:
    root (zarr.Group): The OME-Zarr group containing the label.
    label_name (str): The name of the label to delete.
    """
    if "labels" in root and label_name in root.labels:
        del root.labels[label_name]
        current_labels = root.labels.attrs["labels"]
        new_labels = [lbl for lbl in current_labels if lbl != label_name]
        root.labels.attrs["labels"] = new_labels
    else:
        print(f"Label '{label_name}' does not exist in the OME-Zarr group.")
        return None

# Utility function to read an OME-Zarr group
def read_ome_zarr_group(path):
    """
    Reads an OME-Zarr group from the specified path.
    
    Parameters:
    path (str): The file path to the OME-Zarr group.
    
    Returns:
    zarr.Group: The OME-Zarr group object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path '{path}' does not exist.")
    
    # Open the OME-Zarr group
    store = ozi.parse_url(path, mode="a").store
    root = zarr.group(store=store)
    return root


# Utility function to save label arrays to OME-Zarr format
def save_labels(label, label_name, root):
    # Remove existing label if present to avoid duplicates
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

    Y_dim = root["0"].shape[-2]
    X_dim = root["0"].shape[-1]
    # Write the label array to the OME-Zarr group
    ozw.write_labels(
        labels=label,
        group=root,
        name=label_name,
        axes="tyx",
        scaler=ome_zarr.scale.Scaler(max_layer=1),
        chunks=(1, Y_dim, X_dim),
        storage_options={
            "compressor": zarr.storage.Blosc(cname="zstd", clevel=5),
        },
        metadata={"is_grayscale_label": False},
        delayed=True,
    )