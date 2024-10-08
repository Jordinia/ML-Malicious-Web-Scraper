import csv
from urllib.parse import urlparse

# Function to extract head domain
def get_head_domain(url):
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

# Read domains from CSV file
input_file = 'extracted_domains_warcio.csv'  # Replace with your actual file path
domains = []

with open(input_file, mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        # Assuming the domain is in the first column of each row
        domain = row[0]  
        domains.append(domain)

# Extract head domains and remove duplicates using set
head_domains = list(set(get_head_domain(url) for url in domains))

# Save filtered head domains to a new CSV file
output_file = 'filtered_head_domains.csv'

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    for domain in head_domains:
        csv_writer.writerow([domain])

# Optionally, print filtered domains
for domain in head_domains:
    print(domain)
