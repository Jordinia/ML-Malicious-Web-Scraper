import csv
import sys

csv.field_size_limit(sys.maxsize)

# Path to your large CSV file
csv_file_path = '/mnt/f/SKRIPSI/Data/Processed/domains_and_content_processed.csv'
# Path to the output domains CSV file
domains_csv_file = '/mnt/f/SKRIPSI/Data/Processed/unique_domains2.csv'

# Set to store unique domains
unique_domains = set()

# Initialize a counter for the number of rows
row_count = 0

# Open the input CSV file and extract unique domains
with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    csv_reader = csv.reader((line.replace('\0', '') for line in f))
    
    # Skip the header row if there's one
    next(csv_reader)
    
    # Iterate through each row in the CSV
    for row in csv_reader:
        domain = row[0]  # Assuming domain is in the first column
        
        # Add the domain to the set of unique domains
        unique_domains.add(domain)
        
        # Increment row count
        row_count += 1
        print(str(row_count) + "-" + domain)

# Write the unique domains to a new CSV file
with open(domains_csv_file, 'w', encoding='utf-8', newline='') as f_out:
    csv_writer = csv.writer(f_out)
    
    # Write header
    csv_writer.writerow(['Domain'])
    
    # Write each unique domain to the CSV file
    for domain in unique_domains:
        csv_writer.writerow([domain])

print(f"Unique domains saved to {domains_csv_file}")
print(f"The input CSV file contains {row_count} rows.")