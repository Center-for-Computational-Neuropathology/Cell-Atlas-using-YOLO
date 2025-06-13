import os
import cv2
import glob
import numpy as np
from PIL import Image

def visualize_annotations(input_base_path, output_path, class_labels=None):
    """
    Visualize YOLO annotations on patches and save them to the output directory.
    
    Args:
        input_base_path: Base path containing train/valid/test folders with images and labels
        output_path: Path to save visualized images
        class_labels: Dictionary mapping class IDs to label names (default: {0: "Plaque", 1: "Cluster"})
    """
    if class_labels is None:
        class_labels = {0: "Plaque", 1: "Cluster"}
    
    # Colors for different classes (BGR format)
    class_colors = {
        0: (0, 0, 255),  # Red for plaques
        1: (0, 255, 0)   # Green for clusters
    }
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process each subfolder (train, valid, test)
    for subfolder in ["train", "valid", "test"]:
        images_folder = os.path.join(input_base_path, subfolder, "images")
        labels_folder = os.path.join(input_base_path, subfolder, "labels")
        
        # Skip if folder doesn't exist
        if not os.path.exists(images_folder):
            continue
            
        # Create output subfolder
        output_subfolder = os.path.join(output_path, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Find all images
        for image_path in glob.glob(os.path.join(images_folder, "*.png")):
            # Get base filename
            image_filename = os.path.basename(image_path)
            image_basename = os.path.splitext(image_filename)[0]
            
            # Corresponding label file
            label_path = os.path.join(labels_folder, f"{image_basename}.txt")
            
            # Output path for visualized image
            output_image_path = os.path.join(output_subfolder, image_filename)
            
            # Read image (with PIL and convert to OpenCV format)
            pil_image = Image.open(image_path)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            height, width = cv_image.shape[:2]
            
            # Add patch name to the image
            cv2.putText(cv_image, image_basename, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            # Read and draw annotations if label file exists
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    annotations = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # Valid YOLO format
                            class_id = int(parts[0])
                            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                            
                            # Convert YOLO format to pixel coordinates
                            x_min = int((x_center - bbox_width/2) * width)
                            y_min = int((y_center - bbox_height/2) * height)
                            x_max = int((x_center + bbox_width/2) * width)
                            y_max = int((y_center + bbox_height/2) * height)
                            
                            # Store annotation
                            annotations.append((class_id, x_min, y_min, x_max, y_max))
                    
                    # Draw all annotations on the image
                    for class_id, x_min, y_min, x_max, y_max in annotations:
                        # Get color for this class
                        color = class_colors.get(class_id, (255, 255, 255))
                        
                        # Draw bounding box
                        cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), color, 2)
                        
                        # Get label text
                        label = class_labels.get(class_id, f"Class {class_id}")
                        
                        # Draw label text
                        cv2.putText(cv_image, label, (x_min, y_min - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save the visualized image
            cv2.imwrite(output_image_path, cv_image)
            
    print(f"Visualization complete. Results saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Path to your dataset with YOLO format
    input_path = "/sc/arion/projects/tauomics/Atrophy/YoloV11-train/red_cte_plaques/1024_red_cte_plaques"
    
    # Path to save visualized images
    output_path = "/sc/arion/projects/tauomics/Atrophy/YoloV11-train/red_cte_plaques/1024_red_cte_plaques/1024_red_cte_plaques_visualized"
    
    # Class label mapping (customize as needed)
    class_labels = {
        0: "Plaque",
        1: "Cluster"
    }
    
    # Run visualization
    visualize_annotations(input_path, output_path, class_labels)