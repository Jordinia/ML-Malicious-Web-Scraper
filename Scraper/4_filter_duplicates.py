import csv
import sys
import pandas as pd

# Increase the field size limit to handle large CSV fields
csv.field_size_limit(sys.maxsize)
print(sys.maxsize)

# Paths to the CSV files
domains_and_content_file = '/mnt/f/SKRIPSI/Data/Processed/domains_and_content.csv'
unique_domains_file = '/mnt/f/SKRIPSI/Data/Processed/unique_domains.csv'
output_file = '/mnt/f/SKRIPSI/Data/Processed/domains_and_content_processed.csv'

# Load unique domains and initialize occurrences as 0
unique_domains_df = pd.read_csv(unique_domains_file)
unique_domains_df['Occurrence'] = 0

# Create a dictionary to track the occurrences
domain_occurrences = {domain: 0 for domain in unique_domains_df['Domain'].tolist()}

# Open the output file for writing the processed rows
with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
    writer = csv.writer(output_csv)
    
    # Write the header for the new CSV file
    writer.writerow(['Domain', 'Content'])

    # Open and process the large domains_and_content.csv file in chunks
    with open(domains_and_content_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader((line.replace('\0', '') for line in f))

        # Skip the header row
        next(reader)
        
        # Process each row in domains_and_content.csv
        for row in reader:
            domain = row[0]  # Domain is the first column
            content = row[1]  # Content is the second column

            # If the domain exists in the unique domains list
            if domain in domain_occurrences:
                # Increment the occurrence count
                domain_occurrences[domain] += 1
                print(domain + " " + str(domain_occurrences[domain]))
                
                # Write the row to the new CSV file only if it's the first occurrence
                if domain_occurrences[domain] == 1:
                    writer.writerow([domain, content])

# Update the unique_domains.csv with the occurrence count
unique_domains_df['Occurrence'] = unique_domains_df['Domain'].map(domain_occurrences)
unique_domains_df.to_csv(unique_domains_file, index=False)

print(f"Processed data saved to {output_file}")
