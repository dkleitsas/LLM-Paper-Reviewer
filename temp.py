import csv
import os

# Configuration
input_folder = 'labeled_csvs'
output_folder = 'output_csvs'
target_value = 'APPENDIX'
replacement_value = 'OTHER'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all CSV files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.csv'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row in reader:
                new_row = [replacement_value if cell.strip() == target_value else cell for cell in row]
                print("kaching")
                writer.writerow(new_row)

        print(f"Processed: {filename}")

print(f"\nAll files processed. Modified CSVs saved to: '{output_folder}'")