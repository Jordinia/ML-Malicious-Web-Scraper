import os
import subprocess

# Define the base URL for Common Crawl
base_url = "https://data.commoncrawl.org/"

# Path to your wet.paths file
wet_paths_file = 'wet.paths'

# Define the download location
download_location = '/mnt/f/SKRIPSI/Data'

# Create the download location directory if it doesn't exist
if not os.path.exists(download_location):
    os.makedirs(download_location)

# Function to download files using wget
def download_file(url, download_location):
    # Get the filename from the URL
    filename = url.split('/')[-1]
    output_path = os.path.join(download_location, filename)
    
    # Use wget to download the file
    subprocess.run(['wget', url, '-O', output_path])

# Read the wet.paths file and construct download URLs
with open(wet_paths_file, 'r') as f:
    lines = f.readlines()

# Limit to 50 downloads
for i, line in enumerate(lines[:50]):  # Limit to the first 50 lines
    file_path = line.strip()  # Remove any leading/trailing whitespace
    full_url = base_url + file_path
    
    print(f"Downloading {full_url} to {download_location}...")
    download_file(full_url, download_location)

print("Download complete!")
