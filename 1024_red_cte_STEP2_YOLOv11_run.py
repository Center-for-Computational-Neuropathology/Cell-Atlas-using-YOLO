import os
import cv2
import logging
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
from datetime import datetime

# Configure paths for the new dataset
BASE_DIR = "/sc/arion/projects/tauomics/Atrophy"
RED_CTE_DIR = f"{BASE_DIR}/YoloV11-train/red_cte_plaques/1024_red_cte_plaques"
MODEL_PATH = f"{BASE_DIR}/yolo11x.pt"  # Base model path
DATASET_YAML = f"{RED_CTE_DIR}/dataset.yaml"  # You'll need to create this YAML file
OUTPUT_DIR = f"{RED_CTE_DIR}/runs"
TEST_IMAGES_DIR = f"{RED_CTE_DIR}/test/images"
VISUALIZE_DIR = os.path.join(OUTPUT_DIR, "test_visualizations")
# Create runs folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Create confidence threshold folders
CONF_THRESHOLDS = [
    (0.0, 0.3, "0.0-0.3"),
    (0.3, 0.5, "0.3-0.5"),
    (0.5, 0.7, "0.5-0.7"),
    (0.7, 1.0, "0.7-1.0")
]

# Configure logging with timestamp for unique log files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(OUTPUT_DIR, f"red_cte_yolo_run_{timestamp}.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Create visualization directories
os.makedirs(VISUALIZE_DIR, exist_ok=True)
for low, high, folder_name in CONF_THRESHOLDS:
    conf_dir = os.path.join(VISUALIZE_DIR, folder_name)
    os.makedirs(conf_dir, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_and_print(message, level="info"):
    """Logs and prints a message."""
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)
    print(message)

def get_confidence_folder(confidence):
    """Returns the appropriate folder name based on confidence score."""
    for low, high, folder_name in CONF_THRESHOLDS:
        if low <= confidence < high:
            return folder_name
    return CONF_THRESHOLDS[-1][2]  # Default to highest bin if outside range

def visualize_and_count_detections(model, test_images_dir, visualize_dir):
    """Visualizes bounding boxes and counts plaques and clusters in test images, organized by confidence thresholds."""
    try:
        if not os.path.exists(test_images_dir):
            log_and_print(f"Test images directory not found: {test_images_dir}", level="error")
            raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

        test_images = list(Path(test_images_dir).glob("*.png"))
        if not test_images:
            log_and_print("No images found in the test directory to visualize.", level="warning")
            return

        # Counters for class statistics
        class_counts = {0: 0, 1: 0}  # Class 0: Plaques, Class 1: Clusters
        total_images = len(test_images)
        images_with_detections = 0
        
        # Count by confidence threshold
        confidence_counts = {folder_name: 0 for _, _, folder_name in CONF_THRESHOLDS}

        class_names = {0: "Amyloid Plaque", 1: "Amyloid Cluster"}

        log_and_print(f"Starting processing of {total_images} test images...")

        for i, img_path in enumerate(test_images):
            try:
                # Set a low confidence for prediction to capture all possible detections
                results = model.predict(str(img_path), save=False, conf=0.01)
                
                # Get the original image
                orig_img = cv2.imread(str(img_path))
                if orig_img is None:
                    log_and_print(f"Could not read image: {img_path}", level="warning")
                    continue
                
                # Get detections - fix for YOLO results structure
                boxes = []
                if results and len(results) > 0:
                    boxes = results[0].boxes if hasattr(results[0], "boxes") else []
                
                has_detections = len(boxes) > 0
                
                if has_detections:
                    images_with_detections += 1
                
                # Create a copy of original image for visualization
                img_with_all = orig_img.copy()
                
                # Track which confidence bin this image belongs to (use highest confidence detection)
                max_conf = 0
                
                # Count by class and process each detection
                plaque_count = 0
                cluster_count = 0
                
                for box in boxes:
                    # Get box details
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf)
                    cls = int(box.cls.item())
                    
                    # Count by class (only if above 0.25 confidence)
                    if conf >= 0.25:
                        if cls in class_counts:
                            class_counts[cls] += 1
                        
                        if cls == 0:
                            plaque_count += 1
                        elif cls == 1:
                            cluster_count += 1
                    
                    # Track maximum confidence
                    max_conf = max(max_conf, conf)
                    
                    # Only draw boxes with confidence >= 0.25
                    if conf >= 0.25:
                        # Choose color based on class and confidence
                        if cls == 0:  # Amyloid Plaque
                            color_base = (0, 0, 255)  # Red for plaques (BGR)
                        else:  # Amyloid Cluster
                            color_base = (255, 0, 0)  # Blue for clusters (BGR)
                        
                        # Adjust color intensity based on confidence
                        color_intensity = max(0.4, min(conf, 1.0))  # Scale between 0.4 and 1.0
                        color = tuple(int(c * color_intensity) for c in color_base)
                        
                        # Draw on the all-detections image
                        cv2.rectangle(img_with_all, (x1, y1), (x2, y2), color, 2)
                        class_name = class_names.get(cls, f"Class {cls}")
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(img_with_all, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Generate detailed label with detection info
                confidence_info = f"Plaques: {plaque_count}, Clusters: {cluster_count}, Max Conf: {max_conf:.2f}"
                cv2.putText(img_with_all, confidence_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2)
                
                # Determine output folder based on max confidence
                if has_detections and max_conf >= 0.25:
                    conf_folder = get_confidence_folder(max_conf)
                    confidence_counts[conf_folder] += 1
                    
                    # Save the image with all detections to the confidence folder
                    output_path = os.path.join(visualize_dir, conf_folder, f"{img_path.stem}_viz.png")
                    cv2.imwrite(output_path, img_with_all)
                    
                    log_and_print(f"Image {img_path.name}: {plaque_count} plaques, {cluster_count} clusters detected. Max conf: {max_conf:.2f} → {conf_folder}")
                else:
                    # For images with no detections or low confidence, save to the lowest bin
                    output_path = os.path.join(visualize_dir, CONF_THRESHOLDS[0][2], f"{img_path.stem}_viz.png")
                    cv2.imwrite(output_path, img_with_all)
                    
                    log_and_print(f"Image {img_path.name}: No detections above threshold.")
                
                # Progress indicator
                if (i + 1) % 10 == 0 or (i + 1) == total_images:
                    log_and_print(f"Processed {i + 1}/{total_images} images")
                
            except Exception as e:
                log_and_print(f"Error processing image {img_path}: {e}", level="error")

        # Log summary statistics
        log_and_print("\n=== DETECTION SUMMARY ===")
        log_and_print(f"Total images processed: {total_images}")
        log_and_print(f"Images with detections: {images_with_detections} ({images_with_detections/total_images*100:.1f}%)")
        log_and_print(f"Images without detections: {total_images - images_with_detections} ({(total_images - images_with_detections)/total_images*100:.1f}%)")
        
        for cls_id, count in class_counts.items():
            class_name = class_names.get(cls_id, f"Class {cls_id}")
            log_and_print(f"Total {class_name}s detected: {count}")
            if total_images > 0:
                log_and_print(f"Average {class_name}s per image: {count/total_images:.2f}")
        
        log_and_print(f"\nConfidence distribution:")
        for _, _, folder_name in CONF_THRESHOLDS:
            count = confidence_counts[folder_name]
            percentage = (count / max(1, total_images)) * 100
            log_and_print(f"  {folder_name}: {count} images ({percentage:.1f}%)")
        
        log_and_print(f"Visualized test images organized by confidence in: {visualize_dir}")
        
    except Exception as e:
        log_and_print(f"Error during visualization and detection counting: {e}", level="error")
        raise

def create_dataset_yaml():
    """Creates a dataset.yaml file if it doesn't exist"""
    if os.path.exists(DATASET_YAML):
        log_and_print(f"Dataset YAML exists: {DATASET_YAML}")
        return
    
    log_and_print(f"Creating dataset YAML file: {DATASET_YAML}")
    yaml_content = f"""# Red CTE Dataset YAML
path: {RED_CTE_DIR}  # dataset root directory
train: train/images  # train images
val: valid/images  # validation images
test: test/images  # test images (optional)

# Classes
names:
  0: Amyloid_Plaque
  1: Amyloid_Cluster
"""
    
    try:
        with open(DATASET_YAML, 'w') as f:
            f.write(yaml_content)
        log_and_print("Dataset YAML file created successfully")
    except Exception as e:
        log_and_print(f"Error creating dataset YAML: {e}", level="error")
        raise

if __name__ == '__main__':
    try:
        log_and_print(f"Starting Red CTE YOLO detection script. Log file: {LOG_FILE}")
        
        # Create dataset.yaml if needed
        create_dataset_yaml()
        
        # Initialize YOLO model
        log_and_print("Initializing YOLO model...")
        model = YOLO(MODEL_PATH)
        log_and_print("Model loaded successfully.")

        # Train the model
        log_and_print("Starting training...")
        training_successful = False
        best_model_path = None

        try:
            train_results = model.train(
                data=DATASET_YAML,
                epochs=100,
                batch=16,
                imgsz=640,
                device=0,
                project=OUTPUT_DIR,
                name=f'red_cte_train_{timestamp}',
                exist_ok=True,
                # CHANGE 1: Add checkpoint saving
                save_period=10,  # Save checkpoint every 10 epochs
                # CHANGE 2: Add early stopping
                patience=15  # Stop if no improvement for 15 epochs
            )
            
            # Fixed method to get best model path
            best_model_path = None
            if hasattr(train_results, "save_dir"):
                best_model_path = os.path.join(train_results.save_dir, "weights", "best.pt")
                if os.path.exists(best_model_path):
                    log_and_print(f"Training completed successfully. Best model: {best_model_path}")
                    training_successful = True
                else:
                    log_and_print(f"Best model not found at expected path: {best_model_path}", level="warning")
                    # Try to find the last saved model
                    weights_dir = os.path.join(train_results.save_dir, "weights")
                    if os.path.exists(weights_dir):
                        weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt') and 'epoch' in f]
                        if weight_files:
                            # Sort by epoch number and get the latest
                            weight_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x and x.split('_')[1].split('.')[0].isdigit() else 0, reverse=True)
                            best_model_path = os.path.join(weights_dir, weight_files[0])
                            log_and_print(f"Using latest saved model: {best_model_path}")
                            training_successful = True
                        else:
                            log_and_print("No saved model weights found.", level="error")
                    else:
                        log_and_print(f"Weights directory not found: {weights_dir}", level="error")
            else:
                log_and_print("Training completed but train_results doesn't have expected structure.", level="warning")
                # Try to find the output directory
                run_dir = os.path.join(OUTPUT_DIR, f'red_cte_train_{timestamp}')
                if os.path.exists(run_dir):
                    weight_files = [f for f in os.listdir(run_dir) if f.endswith('.pt') and 'epoch' in f]
                    if weight_files:
                        # Sort by epoch number and get the latest
                        weight_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x and x.split('_')[1].split('.')[0].isdigit() else 0, reverse=True)
                        best_model_path = os.path.join(run_dir, weight_files[0])
                        log_and_print(f"Using latest saved model: {best_model_path}")
                        training_successful = True
                    else:
                        log_and_print("No saved model weights found.", level="error")
                        training_successful = False
                else:
                    log_and_print(f"Training output directory not found: {run_dir}", level="error")
                    training_successful = False
        except Exception as e:
            log_and_print(f"Error during training: {e}", level="error")
            training_successful = False

        # Only proceed with validation and visualization if training was successful
        if training_successful and best_model_path is not None:
            # Check if best model exists before loading
            if not os.path.exists(best_model_path):
                log_and_print(f"Best model not found at {best_model_path}", level="error")
                sys.exit(1)
                
            # Use the best model from training for validation and detection
            try:
                best_model = YOLO(best_model_path)
                log_and_print(f"Loaded best model from {best_model_path}")
                
                # Validate the model
                log_and_print("Starting validation...")
                try:
                    val_results = best_model.val(
                        data=DATASET_YAML,
                        imgsz=640,
                        batch=8,
                        device=0
                    )
                   
                    log_and_print("Validation completed successfully.")

                    # Fixed metrics extraction
                    try:
                        # Try the corrected metrics extraction path
                        metrics = val_results.results_dict if hasattr(val_results, "results_dict") else {}
                        f1_score = metrics.get("metrics/f1", "N/A")
                        precision = metrics.get("metrics/precision", "N/A")
                        recall = metrics.get("metrics/recall", "N/A")
                        mAP50 = metrics.get("metrics/mAP_50", "N/A")
                        mAP50_95 = metrics.get("metrics/mAP_50-95", "N/A")
                        
                        # Fallback to original method if needed
                        if f1_score == "N/A" and hasattr(val_results, "box"):
                            metrics = val_results.box
                            f1_score = metrics.get('f1', 'N/A')
                            precision = metrics.get('precision', 'N/A')
                            recall = metrics.get('recall', 'N/A')
                            mAP50 = metrics.get('map50', 'N/A')
                            mAP50_95 = metrics.get('map', 'N/A')

                        log_and_print(f"Validation Metrics:\nF1 Score: {f1_score}\nPrecision: {precision}\nRecall: {recall}\n"
                                    f"mAP@50: {mAP50}\nmAP@50-95: {mAP50_95}")
                    except Exception as e:
                        log_and_print(f"Skipping metrics extraction due to error: {e}", level="warning")
                except Exception as e:
                    log_and_print(f"Error during validation: {e}", level="error")

                # Visualize bounding boxes on test images and count detections
                log_and_print("Generating visualizations and counting detections for test cohort...")
                visualize_and_count_detections(best_model, TEST_IMAGES_DIR, VISUALIZE_DIR)

                log_and_print("Script execution completed successfully.")

            except Exception as main_e:
                log_and_print(f"Script execution failed: {main_e}", level="error")
                sys.exit(1)
    except Exception as e:
        log_and_print(f"Script execution failed: {e}", level="error")
        sys.exit(1)