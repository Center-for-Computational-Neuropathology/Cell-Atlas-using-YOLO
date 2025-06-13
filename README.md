# Cell-Atlas-using-YOLO-on-Minerva
Order of scripts 
  1.  search_for_WSI_on_Minerva.py --> CP your images to a folder you have access to
  2.  Upload JSONS with annotations (no script)
  3.  1024_red_cte_STEP1_plaque_json_to_patches... --> patch WSI into PNGs and convert labels to YOLO format
  4.  1024_red_cte_STEP1_RECHECK_label_to_image_allignment.py --> Visualize the patches from the previous script          with the YOLO annotations (confirm they align)
  5.  1024_red_cte_STEP2_YOLOv11_run.py --> Model training (adjust necessary parameters and outputs)
