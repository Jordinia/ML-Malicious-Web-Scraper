import pandas as pd

# Load the original CSV
input_file = 'trustpositifkominfo_50000.csv'
data = pd.read_csv(input_file)

# Define the chunk size and the output filenames
chunk_size = 12500
output_files = [
    'trustpositifkominfo_50k_1.csv',
    'trustpositifkominfo_50k_12501.csv',
    'trustpositifkominfo_50k_25001.csv',
    'trustpositifkominfo_50k_37501.csv'
]

# Split the data into chunks and save each chunk as a new CSV file
for i, start_row in enumerate(range(0, len(data), chunk_size)):
    end_row = start_row + chunk_size
    chunk = data.iloc[start_row:end_row]
    chunk.to_csv(output_files[i], index=False)
    print(f"Saved {output_files[i]} with rows {start_row+1} to {end_row}")
