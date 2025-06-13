import os
import json
import random
import openslide
import logging
from PIL import Image
from datetime import datetime

# Configure logging with timestamp for unique log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"/sc/arion/projects/tauomics/Atrophy/amyloid_plaques_scripts/logs/red_cte/step1_{timestamp}.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info(f"Job started: Processing WSI images and annotations. Log file: {log_file}")
print(f"Logging to: {log_file}")

# Define paths
red_cte_image_path = "/sc/arion/projects/tauomics/Atrophy/amyloid_plaques_images/red_CTE"
red_cte_annotation_path = "/sc/arion/projects/tauomics/Atrophy/amyloid_plaques_jsons/red_cte"

# Get all WSI images
wsi_image_paths = [
    os.path.join(red_cte_image_path, file)
    for file in os.listdir(red_cte_image_path)
    if file.endswith(".svs") and os.path.isfile(os.path.join(red_cte_image_path, file))
]

# Get all JSON annotations
annotation_json_paths = [
    os.path.join(red_cte_annotation_path, file)
    for file in os.listdir(red_cte_annotation_path)
    if file.endswith(".json") and os.path.isfile(os.path.join(red_cte_annotation_path, file))
]

# Log detected files
logging.info(f"Found {len(wsi_image_paths)} WSI images.")
logging.info(f"Found {len(annotation_json_paths)} annotation JSON files.")

if not wsi_image_paths or not annotation_json_paths:
    logging.error("No WSI images or annotation JSON files found. Exiting.")
    exit(1)

# Assign YOLO class IDs
CLASS_MAP = {
    "Amyloid_plaques_BTO:0002774": 0,  # Class 0: Amyloid Plaques
    "Amyloid_Plaque_Clusters": 1,  # Class 1: Amyloid Clusters
}

# Convert bounding box to YOLO format
def convert_to_yolo_format(box, img_width, img_height):
    x_min, y_min, x_max, y_max = box
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Create output directories
def prepare_directories(base_path, subfolders=["train", "valid", "test"]):
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_path, subfolder, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_path, subfolder, "labels"), exist_ok=True)

# Save patches and YOLO annotations
def save_patch_and_annotation(patch, annotations, output_folder, patch_name):
    image_folder = os.path.join(output_folder, "images")
    label_folder = os.path.join(output_folder, "labels")

    image_path = os.path.join(image_folder, f"{patch_name}.png")
    patch.save(image_path)

    # Save labels only if annotations exist
    if annotations:
        label_path = os.path.join(label_folder, f"{patch_name}.txt")
        with open(label_path, "w") as f:
            for class_id, yolo_bbox in annotations:
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        logging.info(f"Saving patch with annotations: {image_path}")
    else:
        logging.info(f"Saving patch without annotations: {image_path}")

# Create patches from ROI
def create_ROI_patches(wsi_path, rois, plaques, clusters, wsi_image, img_width, img_height, patch_size, output_base_path, is_test=False):
    patches = []
    image_base_name = os.path.splitext(os.path.basename(wsi_path))[0]

    for roi in rois:
        x_min, y_min, x_max, y_max = map(int, roi)

        # Ensure ROI is within image bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_width, x_max), min(img_height, y_max)

        # Log ROI information
        logging.info(f"Processing ROI: ({x_min}, {y_min}, {x_max}, {y_max}) for {image_base_name}")

        for i in range(y_min, y_max, patch_size):
            for j in range(x_min, x_max, patch_size):
                # Ensure patch does not go outside ROI
                box_x_min = j
                box_y_min = i
                box_x_max = min(j + patch_size, x_max)
                box_y_max = min(i + patch_size, y_max)

                width, height = box_x_max - box_x_min, box_y_max - box_y_min

                # Skip invalid patches or patches that aren't full size
                if width != patch_size or height != patch_size:
                    logging.warning(f"Skipping non-standard patch: size {width}x{height} at ({box_x_min}, {box_y_min})")
                    continue

                patch = wsi_image.read_region((box_x_min, box_y_min), 0, (width, height)).convert("RGB")

                if is_test:
                    patch_name = f"{image_base_name}_test_{box_y_min}_{box_x_min}"
                    patches.append((patch, [], patch_name))  # Empty annotations for test patches
                    logging.info(f"Extracting test patch: {patch_name} at ({box_x_min}, {box_y_min})")
                else:
                    # Process all annotations within this patch
                    patch_annotations = []
                    has_plaque = False
                    has_cluster = False

                    # Check for plaques in this patch
                    for plaque in plaques:
                        p_x_min, p_y_min, p_x_max, p_y_max = plaque
                        
                        # Check if plaque overlaps with current patch
                        if not (p_x_max <= box_x_min or p_x_min >= box_x_max or 
                                p_y_max <= box_y_min or p_y_min >= box_y_max):
                            
                            # Clip annotation to patch boundaries
                            p_x_min = max(p_x_min, box_x_min)
                            p_y_min = max(p_y_min, box_y_min)
                            p_x_max = min(p_x_max, box_x_max)
                            p_y_max = min(p_y_max, box_y_max)

                            # Convert to patch-relative coordinates
                            p_x_min = p_x_min - box_x_min
                            p_y_min = p_y_min - box_y_min
                            p_x_max = p_x_max - box_x_min
                            p_y_max = p_y_max - box_y_min

                            if p_x_min < p_x_max and p_y_min < p_y_max:  # Check if still valid
                                class_id = CLASS_MAP["Amyloid_plaques_BTO:0002774"]
                                yolo_bbox = convert_to_yolo_format([p_x_min, p_y_min, p_x_max, p_y_max], width, height)
                                patch_annotations.append((class_id, yolo_bbox))
                                has_plaque = True

                    # Check for clusters in this patch
                    for cluster in clusters:
                        c_x_min, c_y_min, c_x_max, c_y_max = cluster
                        
                        # Check if cluster overlaps with current patch
                        if not (c_x_max <= box_x_min or c_x_min >= box_x_max or 
                                c_y_max <= box_y_min or c_y_min >= box_y_max):
                            
                            # Clip annotation to patch boundaries
                            c_x_min = max(c_x_min, box_x_min)
                            c_y_min = max(c_y_min, box_y_min)
                            c_x_max = min(c_x_max, box_x_max)
                            c_y_max = min(c_y_max, box_y_max)

                            # Convert to patch-relative coordinates
                            c_x_min = c_x_min - box_x_min
                            c_y_min = c_y_min - box_y_min
                            c_x_max = c_x_max - box_x_min
                            c_y_max = c_y_max - box_y_min

                            if c_x_min < c_x_max and c_y_min < c_y_max:  # Check if still valid
                                class_id = CLASS_MAP["Amyloid_Plaque_Clusters"]
                                yolo_bbox = convert_to_yolo_format([c_x_min, c_y_min, c_x_max, c_y_max], width, height)
                                patch_annotations.append((class_id, yolo_bbox))
                                has_cluster = True

                    # Assign indicators based on presence of annotations
                    if has_plaque and has_cluster:
                        indicator = "P_C"
                    elif has_plaque:
                        indicator = "P"
                    elif has_cluster:
                        indicator = "C"
                    else:
                        indicator = "NA"

                    patch_name = f"{image_base_name}_roi_{box_y_min}_{box_x_min}_{indicator}"
                    patches.append((patch, patch_annotations, patch_name))
                    
                    # Log summary of what's in this patch
                    annotation_summary = f"{len(patch_annotations)} annotations"
                    if patch_annotations:
                        class_counts = {}
                        for class_id, _ in patch_annotations:
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        annotation_summary += f" ({', '.join([f'class {c}: {n}' for c, n in class_counts.items()])})"
                    
                    logging.info(f"Extracting patch: {patch_name} with {annotation_summary}")

    # Save all patches
    if is_test:
        for patch, annotations, patch_name in patches:
            save_patch_and_annotation(patch, annotations, os.path.join(output_base_path, "test"), patch_name)
    else:
        random.shuffle(patches)
        train_count = int(len(patches) * 0.9)

        for idx, (patch, annotations, patch_name) in enumerate(patches):
            folder = "train" if idx < train_count else "valid"
            save_patch_and_annotation(patch, annotations, os.path.join(output_base_path, folder), patch_name)

    return len(patches)

# Process annotations
def process_wsi_annotations(wsi_image_paths, annotation_json_paths, output_base_path, patch_size):
    prepare_directories(output_base_path)
    total_patches = 0
    total_annotations = {"P": 0, "C": 0, "P_C": 0, "NA": 0}

    # Ensure we have matching WSI and annotation files
    paired_files = []
    for wsi_path in wsi_image_paths:
        wsi_base = os.path.splitext(os.path.basename(wsi_path))[0]
        for ann_path in annotation_json_paths:
            ann_base = os.path.splitext(os.path.basename(ann_path))[0]
            if wsi_base in ann_base:  # Match WSI with annotation
                paired_files.append((wsi_path, ann_path))
                break

    logging.info(f"Found {len(paired_files)} matching WSI-annotation pairs")

    for wsi_path, annotation_path in paired_files:
        logging.info(f"Processing: {os.path.basename(wsi_path)} with {os.path.basename(annotation_path)}")
        
        try:
            with open(annotation_path, "r") as f:
                annotations = json.load(f)

            elements = annotations.get("annotation", {}).get("elements", [])
            wsi_image = openslide.OpenSlide(wsi_path)
            img_width, img_height = wsi_image.dimensions

            logging.info(f"WSI dimensions: {img_width}x{img_height}")
            logging.info(f"Found {len(elements)} annotation elements")

            rois, plaques, clusters, rois_to_test = [], [], [], []

            # Process all annotation elements
            for element in elements:
                if element["type"] == "rectangle":
                    try:
                        center = element["center"]
                        width = element["width"]
                        height = element["height"]
                        
                        # Get label and strip whitespace but preserve case
                        label = element.get("label", {}).get("value", "").strip()
                        # Get group if it exists
                        group = element.get("group", "").strip() if "group" in element else ""
                        
                        # For matching purposes, create lowercase versions
                        label_lower = label.lower()
                        group_lower = group.lower()

                        box = [
                            center[0] - width / 2,
                            center[1] - height / 2,
                            center[0] + width / 2,
                            center[1] + height / 2,
                        ]

                        if label_lower == "roi" or group_lower == "roi":
                            rois.append(box)
                        elif label_lower == "roi_to_test" or group_lower == "roi_to_test":
                            rois_to_test.append(box)
                        elif label == "Amyloid_plaques_BTO:0002774":
                            plaques.append(box)
                        elif label == "Amyloid_Plaque_Clusters":
                            clusters.append(box)

                    except KeyError as e:
                        logging.warning(f"Missing key {e} in rectangle element: {element}")

                elif element["type"] == "polyline":
                    logging.info(f"Skipping polyline element: {element.get('id', 'unknown')}")
                else:
                    logging.warning(f"Unknown element type: {element['type']}")

            logging.info(f"Processed annotations: ROIs={len(rois)}, ROI_to_test={len(rois_to_test)}, "
                         f"Plaques={len(plaques)}, Clusters={len(clusters)}")

            # Create patches for train/valid (from ROIs)
            train_patches = 0
            if rois:
                train_patches = create_ROI_patches(wsi_path, rois, plaques, clusters, 
                                                  wsi_image, img_width, img_height, 
                                                  patch_size, output_base_path)
                logging.info(f"Created {train_patches} training/validation patches")
                total_patches += train_patches
            else:
                logging.warning(f"No training ROIs found in {os.path.basename(annotation_path)}")

            # Create patches for test (from ROI_to_test)
            test_patches = 0
            if rois_to_test:
                test_patches = create_ROI_patches(wsi_path, rois_to_test, [], [], 
                                                 wsi_image, img_width, img_height, 
                                                 patch_size, output_base_path, is_test=True)
                logging.info(f"Created {test_patches} test patches")
                total_patches += test_patches
            else:
                logging.warning(f"No testing ROIs found in {os.path.basename(annotation_path)}")

        except Exception as e:
            logging.error(f"Error processing {os.path.basename(wsi_path)}: {str(e)}")

    # Count patch types (based on filenames)
    for subfolder in ["train", "valid"]:
        images_dir = os.path.join(output_base_path, subfolder, "images")
        if os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                if filename.endswith('.png'):
                    if "_P_C_" in filename:
                        total_annotations["P_C"] += 1
                    elif "_P_" in filename:
                        total_annotations["P"] += 1
                    elif "_C_" in filename:
                        total_annotations["C"] += 1
                    elif "_NA_" in filename:
                        total_annotations["NA"] += 1

    # Final statistics
    logging.info(f"Total patches created: {total_patches}")
    logging.info(f"Patch distribution: {total_annotations}")
    return total_patches

if __name__ == "__main__":
    output_path = "/sc/arion/projects/tauomics/Atrophy/YoloV11-train/red_cte_plaques/1024_red_cte_plaques"
    patch_size = 1024
    
    total_patches = process_wsi_annotations(wsi_image_paths, annotation_json_paths, output_path, patch_size)
    
    logging.info(f"Job finished: WSI processing completed. Total patches: {total_patches}")