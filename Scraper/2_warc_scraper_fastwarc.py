# import os
# import gzip
# from fastwarc.warc import ArchiveIterator
# from urllib.parse import urlparse
# import csv

# # Function to extract head domain
# def get_head_domain(url):
#     parsed_url = urlparse(url)
#     return f"{parsed_url.scheme}://{parsed_url.netloc}"

# # Function to extract domains and content using FastWARC from a single WET file
# def extract_domains_and_content_from_wet_fastwarc(file_path):
#     domains_and_content = []
    
#     with gzip.open(file_path, 'rb') as stream:
#         for record in ArchiveIterator(stream):
#             record_type = record.headers.get('WARC-Type')
#             uri = record.headers.get('WARC-Target-URI')
            
#             # Check if it's a valid WARC record type and contains content
#             if record_type in ['conversion', 'response', 'metadata'] and uri:
#                 content = record.reader.read().decode('utf-8', errors='ignore')  # Read content as text
#                 if content:
#                     head_domain = get_head_domain(uri)
#                     domains_and_content.append((head_domain, content))
    
#     return domains_and_content

# # Function to process all .warc.wet.gz files in a directory
# def process_wet_files(directory, output_file):
#     all_records = []
    
#     # Iterate over all .gz files in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith('.warc.wet.gz'):
#             file_path = os.path.join(directory, filename)
#             print(f"Processing {file_path}...")
            
#             # Extract domains and content from each WET file
#             records = extract_domains_and_content_from_wet_fastwarc(file_path)
#             all_records.extend(records)  # Collect all records
            
#     # Save all collected records to a CSV file
#     with open(output_file, 'w', encoding='utf-8') as f_out:
#         csv_writer = csv.writer(f_out)
#         csv_writer.writerow(['Domain', 'Content'])  # Write headers
#         for domain, content in all_records:
#             csv_writer.writerow([domain, content])
    
#     print(f"Domains and content from {len(all_records)} records saved to {output_file}")

# # Specify the directory containing the WET files and the output CSV file
# directory = '/mnt/f/SKRIPSI/Data'
# output_file = 'domains_and_content.csv'

# # Process the WET files and save the result
# process_wet_files(directory, output_file)
import os
import gzip
from fastwarc.warc import ArchiveIterator
from urllib.parse import urlparse
import csv

# Function to extract head domain
def get_head_domain(url):
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

# Function to extract domains and content using FastWARC from a single WET file
def extract_domains_and_content_from_wet_fastwarc(file_path):
    domains_and_content = []
    
    with gzip.open(file_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            record_type = record.headers.get('WARC-Type')
            uri = record.headers.get('WARC-Target-URI')
            
            # Check if it's a valid WARC record type and contains content
            if record_type in ['conversion', 'response', 'metadata'] and uri:
                content = record.reader.read().decode('utf-8', errors='ignore')  # Read content as text
                if content:
                    head_domain = get_head_domain(uri)
                    domains_and_content.append((head_domain, content))
    
    return domains_and_content

# Function to process all .warc.wet.gz files in a directory, writing results incrementally
def process_wet_files(directory, output_file):
    # Ensure that the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    first_file = True  # Flag to check if it's the first file to write headers
    
    # Iterate over all .gz files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.warc.wet.gz'):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}...")
            
            # Extract domains and content from each WET file
            records = extract_domains_and_content_from_wet_fastwarc(file_path)
            
            # Write the extracted records to the CSV file
            with open(output_file, 'w' if first_file else 'a', encoding='utf-8', newline='') as f_out:
                csv_writer = csv.writer(f_out)
                
                # Write headers only for the first file
                if first_file:
                    csv_writer.writerow(['Domain', 'Content'])
                    first_file = False  # Set the flag to False after writing headers
                
                # Write the extracted data to the CSV
                for domain, content in records:
                    csv_writer.writerow([domain, content])
            
            # Free memory by clearing the records list
            del records

    print(f"Processing complete. Data saved to {output_file}.")

# Specify the directory containing the WET files and the output CSV file
directory = '/mnt/f/SKRIPSI/Data'
output_file = '/mnt/f/SKRIPSI/Data/Processed/domains_and_content.csv'

# Process the WET files and save the result
process_wet_files(directory, output_file)
