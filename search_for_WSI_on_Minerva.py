import os

# Updated list of image names to search for
image_names = [
    "NPBB0.20_LHE_B5_HC.svs",
    "NPBB11_LHE_B5_HC.svs"
]

# Directory to search and destination directory
search_directory = "/sc/arion/projects/comppath_500k/neuroFM_slides"
destination_directory = "sc/arion/projects/tauomics/Atrophy/karan/images"
os.makedirs(destination_directory, exist_ok=True)  # Ensure the destination directory exists

# Allowed extensions to filter files
allowed_extensions = {'.tif', '.tiff', '.svs'}  # Add any other extensions as needed

# Function to search for the files
def search_files(directory, filenames):
    found_files = []
    found_filenames = set()  # Track found filenames
    for root, _, files in os.walk(directory):
        for file in files:
            if file in filenames:
                found_files.append(os.path.join(root, file))
                found_filenames.add(file)
    return found_files, found_filenames

# Function to copy files using the `cp` command if not already present in the destination
def copy_files_with_cp(files, destination):
    for file_path in files:
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(destination, file_name)
        # Check if file has an allowed extension
        _, file_extension = os.path.splitext(file_name)
        if file_extension.lower() in allowed_extensions:
            if not os.path.exists(destination_path):
                print(f"Copying {file_name} to {destination} using system-level cp")
                os.system(f'cp "{file_path}" "{destination_path}"')
            else:
                print(f"File {file_name} already exists in {destination}")
        else:
            print(f"Skipping {file_name} due to disallowed extension")

# Perform the search
found_files, found_filenames = search_files(search_directory, image_names)

# Display the results and copy files
if found_files:
    print("Found files:")
    for file_path in found_files:
        print(file_path)
    # Copy the found files
    copy_files_with_cp(found_files, destination_directory)

# Display the not found files
not_found_files = [file for file in image_names if file not in found_filenames]
if not_found_files:
    print("\nFiles not found:")
    for file_name in not_found_files:
        print(file_name)
