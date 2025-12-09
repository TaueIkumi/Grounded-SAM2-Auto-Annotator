#!/bin/bash
IMAGE_DIR="LSMI-images"

for img_path in "$IMAGE_DIR"/Place*_1.jpg; do
    
    if [ -f "$img_path" ]; then
        echo "--------------------------------------------------"
        echo "Processing: $img_path"
        
        python -m src.lsmi_2_pascal_seg --img-path "$img_path"
        
    else
        echo "No files found matching pattern: $IMAGE_DIR/Place*_1.jpg"
    fi

done