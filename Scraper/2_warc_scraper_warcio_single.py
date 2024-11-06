import gzip
import csv
from urllib.parse import urlparse
from warcio.archiveiterator import ArchiveIterator

# Function to extract head domain
def get_head_domain(url):
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

# Function to extract domains and content from .warc.wet.gz using warcio and gzip
def extract_domains_and_content_from_wet_warcio(file_path):
    records = []
    
    # Open the compressed .gz file
    with gzip.open(file_path, 'rb') as stream:
        # Iterate through each record in the WET file
        for record in ArchiveIterator(stream):
            if record.rec_type == 'conversion':  # In WET files, 'conversion' records contain the plain text
                uri = record.rec_headers.get_header('WARC-Target-URI')
                if uri:
                    # Extract domain and content
                    domain = get_head_domain(uri)
                    content = record.content_stream().read().decode('utf-8', errors='replace')  # Handle decoding issues
                    records.append((domain, content))
    
    return records

# Specify the path to your .warc.wet.gz file
file_path = 'CC-MAIN-20240802234508-20240803024508-00000.warc.wet.gz'

# Extract domains and content from the compressed file
records = extract_domains_and_content_from_wet_warcio(file_path)

# Save domains and content to a CSV file
output_file = 'domains_and_content.csv'

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    # Write header
    csv_writer.writerow(['Domain', 'Content'])
    
    # Write domain and content rows
    for domain, content in records:
        csv_writer.writerow([domain, content])

# Optionally, print some results to verify
for domain, content in records[:5]:  # Print the first 5 records
    print(f"Domain: {domain}\nContent: {content[:500]}...\n")

print(f"Domains and content saved to {output_file}")
