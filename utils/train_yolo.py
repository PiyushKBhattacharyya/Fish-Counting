from ultralytics import YOLO
import os

def create_data_yaml(data_dir='yolo_data', output_file='data.yaml'):
    """
    Create a data.yaml file for YOLO training based on the directory structure
    Using kenai-train for training and kenai-val for validation as specified
    """

    # Specifically use kenai-train and kenai-val as requested
    train_dir = os.path.join(data_dir, 'kenai-train')
    val_dir = os.path.join(data_dir, 'kenai-val')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Get all sequence directories in train and val
    train_sequences = []
    val_sequences = []

    for seq in os.listdir(train_dir):
        seq_path = os.path.join(train_dir, seq)
        if os.path.isdir(seq_path) and not seq.startswith('.'):
            train_sequences.append(seq_path)

    for seq in os.listdir(val_dir):
        seq_path = os.path.join(val_dir, seq)
        if os.path.isdir(seq_path) and not seq.startswith('.'):
            val_sequences.append(seq_path)

    # Create data.yaml content
    yaml_content = f"""train: {train_sequences}
val: {val_sequences}

nc: 1
names: ['fish']
"""

    with open(output_file, 'w') as f:
        f.write(yaml_content)

    print(f"Created {output_file} with {len(train_sequences)} train sequences and {len(val_sequences)} val sequences")
    return output_file

def train_yolo_model(data_yaml='data.yaml', epochs=50, imgsz=640):
    """
    Train YOLOv8 model for fish detection
    """
    # Load a pretrained model
    model = YOLO('yolov8n.pt')  # nano model for efficiency

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,  # adjust based on your GPU memory
        name='fish_counter'
    )

    return model

if __name__ == "__main__":
    # Create data configuration
    data_yaml = create_data_yaml()

    # Train the model
    model = train_yolo_model(data_yaml)
    print("Training completed!")