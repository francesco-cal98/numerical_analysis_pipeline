import numpy as np
from pathlib import Path
from typing import Dict, Tuple

def save_to_npz(output_filepath: str | Path, images: np.ndarray, metadata: Dict[str, np.ndarray]) -> None:
    """
    Saves Image data and associated metadata into a compressed .npz file.
    
    Parameters:
    - output_filepath (str | Path): The destination path.
    - images (np.ndarray): A NumPy array containing the visual stimuli.
    - metadata (Dict[str, np.ndarray]): A dictionary of metadata arrays (numerosity, size, etc.).
    """

    # Convert string to Path object for robust path manipulation
    out_path = Path(output_filepath)

    # Create parent directories if they do not exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Unpach dictonary using ** to pass metadata as akeyword arguments
    np.savez_compressed(out_path, images=images, **metadata)
    print(f"Dataset successfully saved to {out_path.resolve()}")

def load_npz_dataset(filepath: str | Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Loads standardized .npz dataset.

    Parameters:
    - filepath (str | Path): Path to .npz file.

    Returns:
    - Tuple[np.ndarray, Dict[str, np.ndarray]]: Images and dictionary of metadata.
    """

    # Convert string to Path object for robust path manipulation
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"The target file {path} does not exist.")
    
    with np.load(path) as data:
        images = data["images"]
        metadata = {key: data[key] for key in data.files if key != "images"}
        return images, metadata