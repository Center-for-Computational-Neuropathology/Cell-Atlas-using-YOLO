# Neuropathological Cell Atlas using YOLO

## Overview

This project implements automated detection and quantification of neuropathological features using [YOLOv11](https://docs.ultralytics.com/models/yolo11/) (You Only Look Once) deep learning architecture. The system is specifically designed to identify and classify key neuropathological markers including **Tau tangles**, **TDP-43 inclusions**, **Lewy bodies**, **Purkinje cells**, **amyloid plaques**, and other critical cellular features in brain tissue samples. This provides a comprehensive solution for automated neuropathological assessment in research and clinical diagnostic applications.

## Features

- **Multi-class Neuropathological Detection**: Simultaneous detection and classification of multiple neuropathological markers
- **Comprehensive Cell Atlas**: Covers major neurodegenerative pathologies and cellular features
- **Real-time Processing**: Fast inference using optimized YOLOv11 architecture
- **Whole Slide Image Support**: Process large brain tissue histopathological images
- **Automated Quantification**: Count and measure pathological features for research analysis
- **Clinical Integration**: Standardized assessment for diagnostic applications
- **Visualization Tools**: View detection results with bounding boxes, confidence scores, and pathology labels

## Repository Structure

```
Neuropathological-Cell-Atlas-YOLO/
├── search_for_WSI_on_Minerva.py      # Locate and process whole slide brain images
├── 1024_red_cte_STEP1_RECHECK_label_to_image_allignment.py  # Pathology label alignment verification
├── 1024_red_cte_STEP1_plaque_json_to_patches.py  # Convert neuropathology annotations to patches
├── 1024_red_cte_STEP2_YOLOv11_run.py  # YOLOv11 training for neuropathological detection
└── README.md                          # Project documentation
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- NumPy
- Matplotlib

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/Center-for-Computational-Neuropathology/Cell-Atlas-using-YOLO.git
cd Cell-Atlas-using-YOLO

# Install required packages
pip install ultralytics opencv-python torch torchvision numpy matplotlib scikit-image

# For neuropathology-specific libraries
pip install openslide-python histomicstk

# For Minerva cluster (if applicable)
module load python/3.8
module load cuda/11.8
module load openslide
```

## Quick Start

### 1. Prepare Brain Tissue Data
```bash
# Search and organize whole slide brain images
python search_for_WSI_on_Minerva.py
```

### 2. Process Neuropathological Annotations
```bash
# Convert pathology annotations to YOLO format and create patches
python 1024_red_cte_STEP1_plaque_json_to_patches.py

# Verify neuropathological label-image alignment
python 1024_red_cte_STEP1_RECHECK_label_to_image_allignment.py
```

### 3. Train the Neuropathology Detection Model
```bash
# Start YOLOv11 training for neuropathological features
python 1024_red_cte_STEP2_YOLOv11_run.py
```

## Workflow

### Step 1: Brain Tissue Data Preparation
- **WSI Processing**: Locate and process Whole Slide Images of brain tissue using `search_for_WSI_on_Minerva.py`
- **Patch Extraction**: Extract relevant patches from large neuropathological images
- **Annotation Conversion**: Convert neuropathological JSON annotations to YOLO-compatible format

### Step 2: Neuropathological Quality Control
- **Alignment Check**: Verify that pathology annotations correctly align with brain tissue patches
- **Data Validation**: Ensure neuropathological data integrity before training
- **Class Balance**: Check distribution of different pathological features

### Step 3: Model Training & Evaluation
- **YOLOv11 Training**: Train the model optimized for neuropathological detection
- **Parameter Tuning**: Adjust hyperparameters for optimal pathology recognition
- **Clinical Validation**: Evaluate model performance against expert neuropathologist annotations

## Usage

### Training a Neuropathology Detection Model
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov11n.pt')  # pretrained model

# Train for neuropathological detection
results = model.train(
    data='neuropathology_dataset.yaml',
    epochs=200,
    imgsz=1024,
    batch=8,
    patience=50,
    save_period=10
)
```

### Running Inference on Brain Tissue
```python
# Load trained neuropathology model
model = YOLO('path/to/neuropathology_model.pt')

# Run inference on brain tissue image
results = model('brain_tissue_section.jpg')

# Get detections for each pathological feature
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"Detected {class_name} with confidence {confidence:.2f}")

# Visualize neuropathological detections
results[0].show()
```

## Dataset Format

The project expects data in YOLO format:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

### dataset.yaml Example
```yaml
path: /path/to/neuropathology/dataset
train: images/train
val: images/val
test: images/test

# Neuropathological Classes
names:
  0: tau_tangles          # Neurofibrillary tangles (Alzheimer's)
  1: amyloid_plaques      # Senile plaques (Alzheimer's)
  2: tdp43_inclusions     # TDP-43 protein aggregates (ALS/FTD)
  3: lewy_bodies          # Alpha-synuclein inclusions (Parkinson's)
  4: purkinje_cells       # Cerebellar Purkinje neurons
  5: reactive_astrocytes  # Activated glial cells
  6: microglial_activation # Activated microglia
  7: neuritic_plaques     # Dystrophic neurites
  8: pick_bodies          # Tau inclusions (Pick's disease)
  9: granulovacuolar_degeneration  # Hippocampal pathology
```

## Configuration

### Model Parameters
- **Image Size**: 1024x1024 pixels (optimized for neuropathological features)
- **Batch Size**: 8-16 (adjustable based on GPU memory and patch complexity)
- **Learning Rate**: Optimized for fine-grained pathological feature detection
- **Epochs**: 200-300 for convergence on neuropathological data
- **Data Augmentation**: Rotation, flipping, color jittering suitable for histopathology

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended for large brain tissue images)
- **RAM**: 32GB+ recommended for whole slide image processing
- **Storage**: 500GB+ for brain tissue datasets and model checkpoints
- **Compute**: Multi-core CPU for image preprocessing and patch extraction

## Results

The neuropathological detection model achieves:
- **High Precision**: Accurate detection of pathological features with minimal false positives
- **Clinical Relevance**: Reliable identification of key neuropathological markers
- **Quantitative Analysis**: Automated counting and measurement of pathological burden
- **Multi-pathology Support**: Simultaneous detection of multiple neurodegenerative markers
- **Fast Inference**: Real-time processing capabilities for clinical workflow integration
- **Scalability**: Handles large whole slide brain images efficiently

### Supported Neuropathological Features
- **Alzheimer's Disease**: Tau tangles, amyloid plaques, neuritic plaques
- **Parkinson's Disease**: Lewy bodies, alpha-synuclein aggregates
- **ALS/FTD**: TDP-43 inclusions, motor neuron pathology
- **Cerebellar Pathology**: Purkinje cell loss and degeneration
- **Inflammatory Markers**: Reactive astrocytes, microglial activation
- **General Neurodegeneration**: Pick bodies, granulovacuolar degeneration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size or use gradient accumulation for large brain images
- **Label alignment errors**: Check neuropathological annotation coordinate systems
- **Class imbalance**: Use weighted loss functions for rare pathological features
- **Slow whole slide processing**: Implement efficient patch extraction and parallel processing
- **False positives on artifacts**: Include tissue artifacts in training data for robust detection

### Neuropathology-Specific Troubleshooting
- **Staining variations**: Train with diverse staining protocols (H&E, immunohistochemistry)
- **Tissue quality**: Handle formalin-fixed paraffin-embedded tissue artifacts
- **Scale variations**: Ensure consistent magnification across training data


**Project Status**: Active Development  
**Last Updated**: September 2025  
**Maintainer**: Center for Computational Neuropathology (Icahn School of Medicine at Mount Sinai)
