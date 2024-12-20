import gzip
from fastwarc.warc import ArchiveIterator
from urllib.parse import urlparse
import csv

# Function to extract head domain
def get_head_domain(url):
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

# Function to extract domains and content using FastWARC
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

# Specify the path to your compressed .warc.wet.gz file
file_path = 'CC-MAIN-20240802234508-20240803024508-00000.warc.wet.gz'

# Extract domains and content
records = extract_domains_and_content_from_wet_fastwarc(file_path)

# Save the records to a CSV file
output_file = 'domains_and_content.csv'
with open(output_file, 'w', encoding='utf-8') as f_out:
    csv_writer = csv.writer(f_out)
    csv_writer.writerow(['Domain', 'Content'])  # Write headers
    for domain, content in records:
        csv_writer.writerow([domain, content])

for domain, content in records[:5]:  # Print the first 5 records
    print(f"Domain: {domain}\nContent: {content[:500]}...\n")

print(f"Domains and content saved to {output_file}")