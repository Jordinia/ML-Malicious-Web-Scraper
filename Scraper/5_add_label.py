import pandas as pd

# Load the CSV file
input_file = "domainhanan_and_content.csv"
df = pd.read_csv(input_file)

# Add a new column "Label" with default values (optional: can be modified as per your logic)
df['Label'] = ''  # Initialize the "Label" column with empty strings or any default value

# Save the updated DataFrame back to a new CSV file
output_file = "domainhanan_and_content_with_labels.csv"
df.to_csv(output_file, index=False)

print("File saved with new 'Label' column.")
