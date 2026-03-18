import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
from typing import Dict, Tuple

# Dinamically add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from groundeep_analysis.core.external_models.datasets.npz_utils import save_to_npz


# Define format-specific parser functions

def parse_mock_data(input_path: Path) -> Tuple[np.ndarray, Dict[str, np.array]]:
    """A dummy parser to test the pipeline without real data."""
    print("Generating mock data...")
    images = np.random.rand(100, 64, 64)
    metadata = {"numerosity": np.random.randint(7, 29, size=100)}
    return images, metadata

def parse_csv_folder(input_path: Path) -> Tuple[np.ndarray, Dict[str, np.array]]:
    """
    Parses a directory containing a 'metadata.csv' and a subfolder of images.
    """

    csv_file = input_path / "metadata.csv"
    img_folder = input_path / "images"

    if not csv_file.exists():
        raise FileNotFoundError(f"Missing metadata.csv in {input_path}")
    if not img_folder.exists():
        raise FileNotFoundError(f"Missing images folder in {input_path}")
    
    print(f"Loading metadata from {csv_file.resolve()}")
    df = pd.read_csv(csv_file)
    
    if "filename" not in df.columns:
        raise ValueError("CSV must contain a 'filename' column to link images to metadata.")
    
    images_list = []
    print("Loading and aligning images")

    for index, row in df.iterrows():
        img_path = img_folder / str(row["filename"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image {img_path} referenced in {csv_file} not found.")
        
        with Image.open(img_path) as img:
            img_array = np.array(img.convert("L"))
            images_list.append(img_array)
    
    images = np.stack(images_list, axis=0)

    metadata = {}
    for col in df.columns:
        if col != "filename":
            metadata[col] = df[col].to_numpy()
    
    return images, metadata

def parse_mat_file(input_path: Path) -> Tuple[np.ndarray, Dict[str, np.array]]:
    """
    Parses a MATLAB .mat file.
    """

    if not input_path.is_file() or input_path.suffix != ".mat":
        raise ValueError(f"Expected a .mat file, got {input_path}")
    
    print(f"Loading MATLAB dataset from {input_path.resolve()}")

    mat_data = sio.loadmat(str(input_path))
    clean_keys = [k for k in mat_data.keys() if not k.startswitch("__")]

    if "images" not in clean_keys:
        raise ValueError("The .mat file must contain a variable named 'images'.")
    
    images = mat_data["images"]
    
    metadata = {}
    for key in clean_keys:
        if key != "images":
            metadata[key] = mat_data[key].flatten()

    return images, metadata
            

# Dispatch dictionary mapping string format arguments to corresponding functions

PARSER = {
    "mock": parse_mock_data,
    "csv": parse_csv_folder,
    "mat": parse_mat_file
    # Add more parsers here
}


# Main execution logic

def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Convert various dataset formats into standard .npz")
    parser.add_argument("--input", type=str, required=True, help="Path to input data (file or folder)")
    parser.add_argument("--output", type=str, required=True, help="Path for output .npz file")
    parser.add_argument("--format", type=str, required=True, choices=PARSER.keys(), help="Format of input data")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Select correct parsing function based on input
    parse_function = PARSER[args.format]

    # Extract raw data into standarized format (NumPy arrays)
    try:
        images, metadata = parse_function(input_path)
    except Exception as e:
        print(f"Error parsing {args.format} data: {e}")
        sys.exit(1)
    
    # Saver to universal .npz format
    save_to_npz(output_path, images, metadata)
    print("Conversion pipeline completed")


if __name__ == "__main__":
    main()