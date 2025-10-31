from ultralytics import YOLO
import cv2
import os
import glob
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def count_fish_in_video(model_path, video_path=None, image_dir=None, output_dir='results', conf_threshold=0.3):
    """
    Count fish in video or image directory using trained YOLO model

    Args:
        model_path: Path to trained YOLO model (.pt file)
        video_path: Path to video file (optional)
        image_dir: Directory containing images (optional)
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
    """
    # Load model
    model = YOLO(model_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results_summary = []

    if video_path:
        # Process video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fish_counts = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run inference
            results = model(frame, conf=conf_threshold)

            # Count fish in current frame
            fish_count = len(results[0].boxes)
            fish_counts.append(fish_count)

            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Save annotated frame (optional, for every 10th frame to save space)
            if frame_count % 10 == 0:
                output_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(output_path, annotated_frame)

        cap.release()

        # Save results
        df = pd.DataFrame({'frame': range(1, len(fish_counts)+1), 'fish_count': fish_counts})
        df.to_csv(os.path.join(output_dir, 'fish_counts.csv'), index=False)

        results_summary.append({
            'source': video_path,
            'total_frames': frame_count,
            'avg_fish_per_frame': sum(fish_counts) / len(fish_counts) if fish_counts else 0,
            'max_fish_in_frame': max(fish_counts) if fish_counts else 0
        })

    elif image_dir:
        # Process image directory
        image_files = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))

        for img_path in image_files:
            # Run inference
            results = model(img_path, conf=conf_threshold)

            # Count fish
            fish_count = len(results[0].boxes)

            results_summary.append({
                'image': os.path.basename(img_path),
                'fish_count': fish_count
            })

            # Save annotated image
            annotated_img = results[0].plot()
            output_path = os.path.join(output_dir, f'annotated_{os.path.basename(img_path)}')
            cv2.imwrite(output_path, annotated_img)

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)

    return results_summary

def analyze_results(results_dir='results'):
    """
    Analyze and visualize fish counting results
    """
    summary_path = os.path.join(results_dir, 'summary.csv')
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

        # Create visualizations
        plt.figure(figsize=(12, 8))

        if 'fish_count' in df.columns and 'image' in df.columns:
            # Bar plot for image counts
            plt.subplot(2, 2, 1)
            sns.barplot(data=df, x='image', y='fish_count')
            plt.xticks(rotation=45, ha='right')
            plt.title('Fish Count per Image')

        elif 'frame' in df.columns:
            # Line plot for video counts
            plt.subplot(2, 2, 1)
            plt.plot(df['frame'], df['fish_count'])
            plt.xlabel('Frame')
            plt.ylabel('Fish Count')
            plt.title('Fish Count Over Time')

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Print statistics
        print("Analysis Results:")
        print(f"Total images/frames: {len(df)}")
        if 'fish_count' in df.columns:
            print(f"Average fish count: {df['fish_count'].mean():.2f}")
            print(f"Maximum fish count: {df['fish_count'].max()}")
            print(f"Minimum fish count: {df['fish_count'].min()}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Count fish using YOLO model')
    parser.add_argument('--model', required=True, help='Path to trained YOLO model')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--images', help='Directory containing images')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')

    args = parser.parse_args()

    if not args.video and not args.images:
        print("Please provide either --video or --images")
        exit(1)

    # Count fish
    results = count_fish_in_video(
        model_path=args.model,
        video_path=args.video,
        image_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf
    )

    print(f"Results saved to {args.output}")

    # Analyze results
    analyze_results(args.output)