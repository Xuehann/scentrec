import os
import csv
import re

# Read all URLs
with open('data/perfumes_2.csv') as f:
    urls = [row[0] for row in csv.reader(f)]

# Get perfume IDs that were saved
saved_ids = set(f[:-5] for f in os.listdir('data/raw_html'))

# Find missing URLs
missing = []
for url in urls:
    match = re.findall(r'/(\d+)-', url)
    if match and match[0] not in saved_ids:
        missing.append(url)

# Write missing ones
with open('data/missing_perfume_urls.csv', 'w') as f:
    writer = csv.writer(f)
    for url in missing:
        writer.writerow([url])

print(f"Found {len(missing)} missing perfume pages.")
