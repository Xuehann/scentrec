import os

raw_dir = 'data/raw_html'
output_file = 'data/rated_perfume_id.csv'

ids = [f[:-5] for f in os.listdir(raw_dir) if f.endswith('.html')]

with open(output_file, 'w') as f:
    for pid in ids:
        f.write(pid + '\n')

print(f"Saved {len(ids)} perfume IDs to {output_file}")
