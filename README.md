# Fish Counting Algorithm

This project implements a fish counting algorithm using YOLOv8 object detection on sonar imagery from the Fish Counting dataset.

## Setup Instructions

### 1. Clone/Download the Repository
Ensure you have the project files in your local directory.

### 2. Set up Virtual Environment

#### Windows (using the provided batch file):
```bash
# Run the setup script
setup_venv.bat
```

#### Manual Setup:
```bash
# Create virtual environment
python -m venv fish_counting_env

# Activate environment
# Windows:
fish_counting_env\Scripts\activate
# Linux/Mac:
# source fish_counting_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (optional, for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Launch Jupyter Notebook
```bash
# With virtual environment activated
jupyter notebook notebook/fish_counting_algorithm.ipynb
```

## Dataset Structure

The dataset contains sonar imagery from different river regions:
- **kenai-train**: Training images from Kenai River
- **kenai-val**: Validation images from Kenai River
- **nushagak**: Images from Nushagak River
- **elwha**: Images from Elwha River
- **kenai-channel**: Images from Kenai River channel
- **kenai-rightbank**: Images from Kenai River right bank

## Algorithm Overview

1. **Data Exploration**: Load and analyze dataset metadata and sample images
2. **Preprocessing**: Prepare data for YOLO training
3. **Model Training**: Configure and train YOLOv8 model for fish detection
4. **Fish Counting**: Implement detection-based counting logic
5. **Visualization**: Display detection results and counting statistics
6. **Evaluation**: Calculate performance metrics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLOv8
- OpenCV
- Pandas, NumPy, Matplotlib
- Jupyter Notebook

## Usage

1. Follow the setup instructions above
2. Open the notebook in Jupyter
3. Run cells sequentially to:
   - Explore the dataset
   - Train/fine-tune the YOLO model
   - Test fish counting on sample clips
   - Visualize results

## Notes

- The notebook includes training code but requires annotated training data
- For production use, train the model on properly labeled fish detection data
- GPU acceleration is recommended for training