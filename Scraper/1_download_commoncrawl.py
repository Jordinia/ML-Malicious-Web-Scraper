import os
import subprocess
import sys

# Define the base URL for Common Crawl
base_url = "https://data.commoncrawl.org/"

# Path to your wet.paths file
paths = 'wet.paths'

# Define the download location
output_path = '/mnt/f/SKRIPSI/Data'

# Create the download location directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Function to download files using wget
def download_file(url, output_path):
    # Get the filename from the URL
    filename = url.split('/')[-1]
    output_path = os.path.join(output_path, filename)
    
    # Use wget to download the file
    subprocess.run(['wget', url, '-O', output_path])

# Check if range arguments were provided
if len(sys.argv) != 3:
    print("Usage: python download_commoncrawl.py <start_line> <end_line>")
    sys.exit(1)

# Parse the start and end lines from the command-line arguments
start_line = int(sys.argv[1])
end_line = int(sys.argv[2])

# Ensure valid range
if start_line < 1 or end_line < start_line:
    print("Error: Invalid line range.")
    sys.exit(1)

# Read the wet.paths file and construct download URLs
with open(paths, 'r') as f:
    lines = f.readlines()

# Ensure the end line does not exceed the number of lines in the file
if end_line > len(lines):
    print(f"Error: The end line {end_line} exceeds the number of lines in the file ({len(lines)}).")
    sys.exit(1)

# Download the files in the specified range (adjusting for zero-indexing)
for i, line in enumerate(lines[start_line-1:end_line]):
    file_path = line.strip()  # Remove any leading/trailing whitespace
    full_url = base_url + file_path
    
    print(f"Downloading {full_url} to {output_path}...")
    download_file(full_url, output_path)

print("Download complete!")
