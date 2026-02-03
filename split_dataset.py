import zipfile
import os

# Config
SOURCE_ZIP = "benchmarks/deepseek_ising_dataset.zip"
PART1_NAME = "benchmarks/ising_benchmark_part1.zip"
PART2_NAME = "benchmarks/ising_benchmark_part2.zip"

def split_zip():
    if not os.path.exists(SOURCE_ZIP):
        print(f"Error: {SOURCE_ZIP} not found.")
        return

    print(f"Reading {SOURCE_ZIP}...")
    with zipfile.ZipFile(SOURCE_ZIP, 'r') as src:
        file_list = src.namelist()
        midpoint = len(file_list) // 2
        
        files1 = file_list[:midpoint]
        files2 = file_list[midpoint:]

        print(f"Creating Part 1 with {len(files1)} files...")
        with zipfile.ZipFile(PART1_NAME, 'w', zipfile.ZIP_DEFLATED) as z1:
            for f in files1:
                z1.writestr(f, src.read(f))

        print(f"Creating Part 2 with {len(files2)} files...")
        with zipfile.ZipFile(PART2_NAME, 'w', zipfile.ZIP_DEFLATED) as z2:
            for f in files2:
                z2.writestr(f, src.read(f))

    print("Removing original large file...")
    os.remove(SOURCE_ZIP)
    print("Success! Split into part1.zip and part2.zip")

if __name__ == "__main__":
    split_zip()  # FIXED: This matches the function name now