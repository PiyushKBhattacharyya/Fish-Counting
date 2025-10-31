import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import cv2
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler

def analyze_dataset_structure(data_dir='Data/tiny_dataset', yolo_dir='yolo_data'):
    """
    Comprehensive analysis of the fish counting dataset structure and statistics
    """
    print("=" * 60)
    print("COMPREHENSIVE FISH COUNTING DATASET ANALYSIS")
    print("=" * 60)

    # Original dataset analysis
    print("\n1. ORIGINAL DATASET STRUCTURE ANALYSIS")
    print("-" * 40)

    locations = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    dataset_stats = []

    for location in locations:
        print(f"\nAnalyzing location: {location.upper()}")
        loc_path = os.path.join(data_dir, location)

        if location == 'annotations-tiny':
            subdirs = [d for d in os.listdir(loc_path) if os.path.isdir(os.path.join(loc_path, d))]
            for subdir in subdirs:
                sub_path = os.path.join(loc_path, subdir)
                sequences = [s for s in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, s))]
                print(f"  └── {subdir}: {len(sequences)} video sequences")

        elif location == 'raw':
            subdirs = [d for d in os.listdir(loc_path) if os.path.isdir(os.path.join(loc_path, d))]
            for subdir in subdirs:
                sub_path = os.path.join(loc_path, subdir)
                sequences = [s for s in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, s))]
                print(f"  └── {subdir}: {len(sequences)} video sequences")

                # Analyze image properties
                if sequences:
                    first_seq = os.path.join(sub_path, sequences[0])
                    images = [f for f in os.listdir(first_seq) if f.endswith(('.jpg', '.png'))]
                    if images:
                        img_path = os.path.join(first_seq, images[0])
                        img = cv2.imread(img_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            print(f"      ├── Sample resolution: {w}×{h} pixels")
                            print(f"      └── Color channels: {img.shape[2] if len(img.shape) > 2 else 1}")

    # YOLO dataset analysis
    print("\n2. YOLO DATASET STRUCTURE ANALYSIS")
    print("-" * 40)

    yolo_locations = [d for d in os.listdir(yolo_dir) if os.path.isdir(os.path.join(yolo_dir, d)) and not d.startswith('.')]

    total_sequences = 0
    total_images = 0
    total_annotations = 0
    sequence_lengths = []

    for location in yolo_locations:
        print(f"\nLocation: {location.upper()}")
        loc_path = os.path.join(yolo_dir, location)

        sequences = [s for s in os.listdir(loc_path) if os.path.isdir(os.path.join(loc_path, s))]
        print(f"├── Sequences: {len(sequences)}")

        seq_images = 0
        seq_annotations = 0

        for seq in sequences:
            seq_path = os.path.join(loc_path, seq)

            # Count images and annotations recursively (for nested subfolders)
            images = []
            annotations = []
            for root, dirs, files in os.walk(seq_path):
                for file in files:
                    if file.endswith('.txt'):
                        annotations.append(os.path.join(root, file))

            seq_annotations += len(annotations)

        # Count images from corresponding raw data location
        raw_loc_path = os.path.join(data_dir, 'raw', location)
        if os.path.exists(raw_loc_path):
            for root, dirs, files in os.walk(raw_loc_path):
                for file in files:
                    if file.endswith(('.jpg', '.png')):
                        seq_images += 1

        sequence_lengths.append(seq_images)

        print(f"├── Images: {seq_images}")
        print(f"└── Annotations: {seq_annotations}")

        total_sequences += len(sequences)
        total_images += seq_images
        total_annotations += seq_annotations

        dataset_stats.append({
            'location': location,
            'sequences': len(sequences),
            'images': seq_images,
            'annotations': seq_annotations,
            'avg_seq_length': seq_images / len(sequences) if sequences else 0
        })

    print("\n3. DATASET SUMMARY STATISTICS")
    print("-" * 40)
    print(f"├── Total locations: {len(yolo_locations)}")
    print(f"├── Total sequences: {total_sequences}")
    print(f"├── Total images: {total_images:,}")
    print(f"├── Total annotations: {total_annotations:,}")
    print(".1f")
    print(".1f")
    print(f"├── Images per annotation ratio: {total_images/total_annotations:.2f}")

    return pd.DataFrame(dataset_stats)

def analyze_annotations(yolo_dir='yolo_data'):
    """
    Analyze annotation statistics (bounding boxes, classes, etc.)
    """
    print("\n=== Annotation Analysis ===\n")

    all_boxes = []
    sequence_stats = []

    locations = [d for d in os.listdir(yolo_dir) if os.path.isdir(os.path.join(yolo_dir, d)) and not d.startswith('.')]

    for location in locations:
        loc_path = os.path.join(yolo_dir, location)

        sequences = [s for s in os.listdir(loc_path) if os.path.isdir(os.path.join(loc_path, s))]

        for seq in sequences:
            seq_path = os.path.join(loc_path, seq)
            annotations = [f for f in os.listdir(seq_path) if f.endswith('.txt')]

            seq_boxes = 0
            seq_fish_counts = []

            for ann_file in annotations:
                ann_path = os.path.join(seq_path, ann_file)

                with open(ann_path, 'r') as f:
                    lines = f.readlines()

                fish_count = len(lines)
                seq_fish_counts.append(fish_count)
                seq_boxes += fish_count

                # Parse bounding boxes
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])

                        all_boxes.append({
                            'location': location,
                            'sequence': seq,
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })

            sequence_stats.append({
                'location': location,
                'sequence': seq,
                'total_boxes': seq_boxes,
                'avg_fish_per_frame': np.mean(seq_fish_counts) if seq_fish_counts else 0,
                'max_fish_per_frame': max(seq_fish_counts) if seq_fish_counts else 0
            })

    # Convert to DataFrames
    boxes_df = pd.DataFrame(all_boxes)
    seq_stats_df = pd.DataFrame(sequence_stats)

    # Summary statistics
    print("Bounding Box Statistics:")
    print(f"Total bounding boxes: {len(boxes_df)}")
    print(f"Unique classes: {boxes_df['class_id'].nunique()}")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    print("\nFish Count Statistics:")
    print(".2f")
    print(f"Maximum fish per frame: {seq_stats_df['max_fish_per_frame'].max()}")

    return boxes_df, seq_stats_df

def create_visualizations(dataset_stats, boxes_df, seq_stats_df, output_dir='EDA'):
    """
    Create comprehensive visualizations for ML research paper on fish counting
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set professional style
    plt.style.use('default')
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Figure 1: Dataset Overview - Professional Multi-panel
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Fish Counting Dataset: Comprehensive Analysis for ML Research',
                 fontsize=16, fontweight='bold', y=0.95)

    # 1. Dataset distribution by location
    ax1 = fig.add_subplot(gs[0, :2])
    locations = dataset_stats['location']
    images = dataset_stats['images']
    annotations = dataset_stats['annotations']

    x = np.arange(len(locations))
    width = 0.35

    ax1.bar(x - width/2, images, width, label='Images', alpha=0.8, color='#2E86AB')
    ax1.bar(x + width/2, annotations, width, label='Annotations', alpha=0.8, color='#A23B72')

    ax1.set_xlabel('Geographic Location', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Dataset Size Distribution by Location', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([loc.title() for loc in locations], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Sequence length distribution
    ax2 = fig.add_subplot(gs[0, 2])
    seq_lengths = dataset_stats['avg_seq_length']
    ax2.hist(seq_lengths, bins=10, alpha=0.7, color='#F18F01', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Average Frames per Sequence', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Sequence Length Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Fish count distribution (log scale for better visibility)
    ax3 = fig.add_subplot(gs[1, :2])
    fish_counts = seq_stats_df['avg_fish_per_frame']
    bins = np.logspace(np.log10(max(0.1, fish_counts.min())), np.log10(fish_counts.max()), 20)

    counts, bin_edges = np.histogram(fish_counts, bins=bins)
    ax3.hist(fish_counts, bins=bins, alpha=0.7, color='#C73E1D', edgecolor='black', linewidth=0.5)
    ax3.set_xscale('log')
    ax3.set_xlabel('Average Fish Count per Frame (log scale)', fontsize=12)
    ax3.set_ylabel('Number of Sequences', fontsize=12)
    ax3.set_title('Fish Density Distribution Across Sequences', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add statistical annotations
    mean_fish = fish_counts.mean()
    median_fish = fish_counts.median()
    ax3.axvline(mean_fish, color='blue', linestyle='--', alpha=0.7, label='.1f')
    ax3.axvline(median_fish, color='green', linestyle='--', alpha=0.7, label='.1f')
    ax3.legend()

    # 4. Bounding box analysis
    ax4 = fig.add_subplot(gs[1, 2])
    bbox_areas = boxes_df['width'] * boxes_df['height']
    ax4.hist(bbox_areas, bins=30, alpha=0.7, color='#6B73A6', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Bounding Box Area (normalized)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Object Size Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Correlation analysis
    ax5 = fig.add_subplot(gs[2, :])
    # Scatter plot: sequence length vs average fish count
    seq_length_fish = pd.merge(dataset_stats, seq_stats_df.groupby('location')['avg_fish_per_frame'].mean().reset_index(),
                              on='location', how='left')

    ax5.scatter(seq_length_fish['avg_seq_length'], seq_length_fish['avg_fish_per_frame'],
               s=100, alpha=0.7, color='#3A7D44', edgecolors='black', linewidth=0.5)

    for i, loc in enumerate(seq_length_fish['location']):
        ax5.annotate(loc.title(), (seq_length_fish['avg_seq_length'][i], seq_length_fish['avg_fish_per_frame'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax5.set_xlabel('Average Frames per Sequence', fontsize=12)
    ax5.set_ylabel('Average Fish Count per Frame', fontsize=12)
    ax5.set_title('Correlation: Sequence Length vs Fish Density', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Calculate and display correlation
    corr = seq_length_fish['avg_seq_length'].corr(seq_length_fish['avg_fish_per_frame'])
    ax5.text(0.05, 0.95, '.3f', transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_dataset_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2: Statistical Summary Dashboard
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Statistical Summary Dashboard', fontsize=16, fontweight='bold')

    # Box plot of fish counts by location
    ax = axes[0, 0]
    sns.boxplot(data=seq_stats_df, x='location', y='avg_fish_per_frame', ax=ax, palette='Set3')
    ax.set_title('Fish Count Distribution by Location', fontweight='bold')
    ax.set_xlabel('Location')
    ax.set_ylabel('Avg Fish per Frame')
    ax.tick_params(axis='x', rotation=45)

    # Violin plot of bounding box areas
    ax = axes[0, 1]
    sns.violinplot(data=boxes_df, y='width', ax=ax, color='#87CEEB', inner='quartile')
    ax.set_title('Bounding Box Width Distribution', fontweight='bold')
    ax.set_ylabel('Normalized Width')

    # Sequence statistics
    ax = axes[0, 2]
    seq_stats = ['Total Sequences', 'Total Images', 'Total Annotations', 'Avg Sequence Length']
    seq_values = [dataset_stats['sequences'].sum(),
                 dataset_stats['images'].sum(),
                 dataset_stats['annotations'].sum(),
                 dataset_stats['avg_seq_length'].mean()]

    bars = ax.bar(range(len(seq_stats)), seq_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_title('Dataset Statistics Overview', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(seq_stats)))
    ax.set_xticklabels(seq_stats, rotation=45, ha='right')

    # Add value labels on bars
    for bar, value in zip(bars, seq_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{int(value):,}', ha='center', va='bottom', fontweight='bold')

    # Fish count histogram with KDE
    ax = axes[1, 0]
    sns.histplot(data=seq_stats_df, x='avg_fish_per_frame', ax=ax, kde=True, color='#FF8C42', alpha=0.7)
    ax.set_title('Fish Count per Frame Distribution', fontweight='bold')
    ax.set_xlabel('Average Fish Count')
    ax.set_ylabel('Frequency')

    # Bounding box aspect ratio
    ax = axes[1, 1]
    aspect_ratios = boxes_df['width'] / boxes_df['height']
    sns.histplot(aspect_ratios, ax=ax, bins=30, color='#6A5ACD', alpha=0.7)
    ax.set_title('Bounding Box Aspect Ratio Distribution', fontweight='bold')
    ax.set_xlabel('Width/Height Ratio')
    ax.set_ylabel('Frequency')

    # Cumulative distribution
    ax = axes[1, 2]
    sorted_counts = np.sort(seq_stats_df['avg_fish_per_frame'])
    yvals = np.arange(len(sorted_counts))/float(len(sorted_counts)-1)
    ax.plot(sorted_counts, yvals, color='#32CD32', linewidth=2)
    ax.fill_between(sorted_counts, yvals, alpha=0.3, color='#32CD32')
    ax.set_title('Cumulative Distribution of Fish Counts', fontweight='bold')
    ax.set_xlabel('Average Fish Count per Frame')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 3: Research Insights
    fig3, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig3.suptitle('Key Research Insights for Fish Counting ML Model', fontsize=16, fontweight='bold')

    # Location-wise performance prediction
    ax = axes[0]
    loc_performance = seq_stats_df.groupby('location').agg({
        'avg_fish_per_frame': ['mean', 'std', 'count']
    }).round(2)

    loc_performance.columns = ['mean_fish', 'std_fish', 'sample_size']
    loc_performance = loc_performance.reset_index()

    # Calculate expected difficulty (higher fish count = harder)
    loc_performance['difficulty_score'] = loc_performance['mean_fish'] * (1 + loc_performance['std_fish'])

    bars = ax.barh(loc_performance['location'], loc_performance['difficulty_score'],
                  color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(loc_performance))))
    ax.set_title('Predicted Model Difficulty by Location', fontweight='bold', fontsize=14)
    ax.set_xlabel('Difficulty Score (higher = more challenging)')
    ax.set_ylabel('Location')

    # Add value labels
    for bar, score in zip(bars, loc_performance['difficulty_score']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               '.2f', ha='left', va='center', fontweight='bold')

    # Model requirements analysis
    ax = axes[1]

    # Calculate dataset characteristics for model selection
    total_samples = dataset_stats['annotations'].sum()
    avg_objects_per_image = seq_stats_df['avg_fish_per_frame'].mean()
    bbox_size_variation = (boxes_df['width'] * boxes_df['height']).std()

    metrics = ['Total Training Samples', 'Avg Objects/Image', 'Object Size Variation', 'Location Diversity']
    values = [total_samples, avg_objects_per_image, bbox_size_variation, len(dataset_stats)]
    recommended = ['Sufficient', 'Medium Density', 'High Variation', 'Good Diversity']

    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, values, color=['#4CAF50', '#FF9800', '#F44336', '#2196F3'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_title('Dataset Characteristics & Model Implications', fontweight='bold', fontsize=14)
    ax.set_xlabel('Value')

    # Add recommendation text
    for i, (bar, rec) in enumerate(zip(bars, recommended)):
        ax.text(bar.get_width() + max(values)*0.02, bar.get_y() + bar.get_height()/2,
               rec, ha='left', va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'research_insights.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Save individual plots separately
    print(f"\nSaving individual plots to {output_dir}/:")

    # Figure 1: Dataset Distribution
    plt.figure(figsize=(12, 8))
    x = np.arange(len(locations))
    width = 0.35
    plt.bar(x - width/2, images, width, label='Images', alpha=0.8, color='#2E86AB')
    plt.bar(x + width/2, annotations, width, label='Annotations', alpha=0.8, color='#A23B72')
    plt.xlabel('Geographic Location', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Dataset Size Distribution by Location', fontsize=14, fontweight='bold')
    plt.xticks(x, [loc.title() for loc in locations], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Sequence Length Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(seq_lengths, bins=10, alpha=0.7, color='#F18F01', edgecolor='black', linewidth=0.5)
    plt.xlabel('Average Frames per Sequence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Sequence Length Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sequence_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Fish Count Distribution
    plt.figure(figsize=(10, 6))
    fish_counts = seq_stats_df['avg_fish_per_frame']
    bins = np.logspace(np.log10(max(0.1, fish_counts.min())), np.log10(fish_counts.max()), 20)
    plt.hist(fish_counts, bins=bins, alpha=0.7, color='#C73E1D', edgecolor='black', linewidth=0.5)
    plt.xscale('log')
    plt.xlabel('Average Fish Count per Frame (log scale)', fontsize=12)
    plt.ylabel('Number of Sequences', fontsize=12)
    plt.title('Fish Density Distribution Across Sequences', fontsize=14, fontweight='bold')
    mean_fish = fish_counts.mean()
    median_fish = fish_counts.median()
    plt.axvline(mean_fish, color='blue', linestyle='--', alpha=0.7, label='.1f')
    plt.axvline(median_fish, color='green', linestyle='--', alpha=0.7, label='.1f')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fish_count_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 4: Bounding Box Analysis
    plt.figure(figsize=(8, 6))
    bbox_areas = boxes_df['width'] * boxes_df['height']
    plt.hist(bbox_areas, bins=30, alpha=0.7, color='#6B73A6', edgecolor='black', linewidth=0.5)
    plt.xlabel('Bounding Box Area (normalized)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Object Size Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 5: Correlation Analysis
    plt.figure(figsize=(10, 6))
    seq_length_fish = pd.merge(dataset_stats, seq_stats_df.groupby('location')['avg_fish_per_frame'].mean().reset_index(),
                              on='location', how='left')
    plt.scatter(seq_length_fish['avg_seq_length'], seq_length_fish['avg_fish_per_frame'],
               s=100, alpha=0.7, color='#3A7D44', edgecolors='black', linewidth=0.5)
    for i, loc in enumerate(seq_length_fish['location']):
        plt.annotate(loc.title(), (seq_length_fish['avg_seq_length'][i], seq_length_fish['avg_fish_per_frame'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.xlabel('Average Frames per Sequence', fontsize=12)
    plt.ylabel('Average Fish Count per Frame', fontsize=12)
    plt.title('Correlation: Sequence Length vs Fish Density', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    corr = seq_length_fish['avg_seq_length'].corr(seq_length_fish['avg_fish_per_frame'])
    plt.text(0.05, 0.95, '.3f', transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 6: Box Plot by Location
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=seq_stats_df, x='location', y='avg_fish_per_frame', palette='Set3')
    plt.title('Fish Count Distribution by Location', fontweight='bold', fontsize=14)
    plt.xlabel('Location', fontsize=12)
    plt.ylabel('Avg Fish per Frame', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fish_count_by_location.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 7: Aspect Ratio Distribution
    plt.figure(figsize=(8, 6))
    aspect_ratios = boxes_df['width'] / boxes_df['height']
    plt.hist(aspect_ratios, bins=30, alpha=0.7, color='#6A5ACD', edgecolor='black', linewidth=0.5)
    plt.xlabel('Width/Height Ratio', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Bounding Box Aspect Ratio Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aspect_ratio_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 8: Cumulative Distribution
    plt.figure(figsize=(8, 6))
    sorted_counts = np.sort(seq_stats_df['avg_fish_per_frame'])
    yvals = np.arange(len(sorted_counts))/float(len(sorted_counts)-1)
    plt.plot(sorted_counts, yvals, color='#32CD32', linewidth=2)
    plt.fill_between(sorted_counts, yvals, alpha=0.3, color='#32CD32')
    plt.xlabel('Average Fish Count per Frame', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution of Fish Counts', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 9: Location Difficulty Analysis
    plt.figure(figsize=(10, 6))
    loc_performance = seq_stats_df.groupby('location').agg({
        'avg_fish_per_frame': ['mean', 'std', 'count']
    }).round(2)
    loc_performance.columns = ['mean_fish', 'std_fish', 'sample_size']
    loc_performance = loc_performance.reset_index()
    loc_performance['difficulty_score'] = loc_performance['mean_fish'] * (1 + loc_performance['std_fish'])

    bars = plt.barh(loc_performance['location'], loc_performance['difficulty_score'],
                   color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(loc_performance))))
    plt.xlabel('Difficulty Score (higher = more challenging)', fontsize=12)
    plt.ylabel('Location', fontsize=12)
    plt.title('Predicted Model Difficulty by Location', fontsize=14, fontweight='bold')
    for bar, score in zip(bars, loc_performance['difficulty_score']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               '.2f', ha='left', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'location_difficulty.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Individual plots saved:")
    print("├── dataset_distribution.png")
    print("├── sequence_length_distribution.png")
    print("├── fish_count_distribution.png")
    print("├── bbox_analysis.png")
    print("├── correlation_analysis.png")
    print("├── fish_count_by_location.png")
    print("├── aspect_ratio_distribution.png")
    print("├── cumulative_distribution.png")
    print("└── location_difficulty.png")

def export_statistics(dataset_stats, boxes_df, seq_stats_df, output_dir='EDA'):
    """
    Export statistical summaries for research paper
    """
    # Dataset summary
    summary = {
        'total_locations': len(dataset_stats),
        'total_sequences': dataset_stats['sequences'].sum(),
        'total_images': dataset_stats['images'].sum(),
        'total_annotations': dataset_stats['annotations'].sum(),
        'avg_fish_per_frame': seq_stats_df['avg_fish_per_frame'].mean(),
        'max_fish_per_frame': seq_stats_df['max_fish_per_frame'].max(),
        'bbox_area_mean': (boxes_df['width'] * boxes_df['height']).mean(),
        'bbox_area_std': (boxes_df['width'] * boxes_df['height']).std()
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'dataset_summary.csv'), index=False)

    # Export detailed statistics
    dataset_stats.to_csv(os.path.join(output_dir, 'location_stats.csv'), index=False)
    seq_stats_df.to_csv(os.path.join(output_dir, 'sequence_stats.csv'), index=False)

    # Print summary for console
    print("\n=== Dataset Summary for Research Paper ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(".3f")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Analyze dataset
    dataset_stats = analyze_dataset_structure()

    # Analyze annotations
    boxes_df, seq_stats_df = analyze_annotations()

    # Create visualizations
    create_visualizations(dataset_stats, boxes_df, seq_stats_df)

    # Export statistics
    export_statistics(dataset_stats, boxes_df, seq_stats_df)

    print("\nEDA analysis completed! Check the EDA folder for visualizations and statistics.")