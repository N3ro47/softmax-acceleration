import numpy as np
import argparse
import os

def generate_data(vector_size, file_path):
    print(f"Generating data for size {vector_size} at {file_path}...")
    data = np.random.uniform(-10.0, 10.0, vector_size).astype(np.float32)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.tofile(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary test data.")
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    generate_data(args.size, args.output)
