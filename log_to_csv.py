import re
import csv

# Input file
input_file = "training_logs/raw_data/tl1.txt"

# Output file
output_file = "training_logs/scraped_data/training_metrics1.csv"

# Read raw log
with open(input_file, "r") as f:
    text = f.read()

# Extract metrics
maps = re.findall(r"Train mAP: ([0-9.eE+-]+)", text)
losses = re.findall(r"Mean loss was ([0-9.eE+-]+)", text)

print(f"Found {len(maps)} mAP values")
print(f"Found {len(losses)} loss values")

# Write CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["epoch", "mAP", "loss"])

    for epoch, (m, l) in enumerate(zip(maps, losses), start=1):
        writer.writerow([epoch, float(m), float(l)])

print(f"Saved to {output_file}")