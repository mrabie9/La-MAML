import numpy as np
import sys

def combine_npz(file1, file2, output_file):
    # Load both .npz files
    data1 = np.load(file1)
    data2 = np.load(file2)

    # Combine the two dictionaries
    combined_data = {**data1, **data2}

    # Save combined data into a new .npz file
    np.savez(output_file, **combined_data)
    print(f"Combined .npz saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_npz.py file1.npz file2.npz output.npz")
        sys.exit(1)

    combine_npz(sys.argv[1], sys.argv[2], sys.argv[3])
