import os
import pandas as pd

# Read the stats
loc_stats = pd.read_csv('EDA/location_stats.csv')
seq_stats = pd.read_csv('EDA/sequence_stats.csv')

print('=== DATA CONSISTENCY CHECK ===')
print()

# Check for missing annotations
print('1. Missing Annotations Check:')
missing_ann = loc_stats[loc_stats['annotations'] < loc_stats['images']]
if not missing_ann.empty:
    print('WARNING: Found sequences with missing annotations:')
    for _, row in missing_ann.iterrows():
        print(f'  - {row["location"]}: {row["images"]} images vs {row["annotations"]} annotations')
else:
    print('✓ All locations have sufficient annotations')

print()

# Check image-to-annotation ratio
print('2. Image-to-Annotation Ratio Analysis:')
loc_stats['img_ann_ratio'] = loc_stats['images'] / loc_stats['annotations']
print(loc_stats[['location', 'images', 'annotations', 'img_ann_ratio']].round(3))

print()

# Check for empty sequences
print('3. Empty Sequence Check:')
empty_seq = seq_stats[seq_stats['total_boxes'] == 0]
if not empty_seq.empty:
    print('WARNING: Found empty sequences:')
    for _, row in empty_seq.iterrows():
        print(f'  - {row["location"]}/{row["sequence"]}: 0 annotations')
else:
    print('✓ No empty sequences found')

print()

# Check location distribution
print('4. Location Distribution:')
print(loc_stats.groupby('location')['sequences'].sum())

print()

# Check fish count statistics
print('5. Fish Count Statistics by Location:')
fish_stats = seq_stats.groupby('location')['avg_fish_per_frame'].describe()
print(fish_stats.round(3))