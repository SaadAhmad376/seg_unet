import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def convert_coco_to_mask(coco_annotation_file, image_dir, output_dir):

    coco = COCO(coco_annotation_file)
    os.makedirs(output_dir, exist_ok=True)

    img_ids = coco.getImgIds()
    
    for img_id in img_ids:
        # Load image information
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
                
            if 'segmentation' in ann:
                if ann['category_id'] == 1:
                    if isinstance(ann['segmentation'], list):
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((len(seg) // 2, 2))
                            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                    else:
                        rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
                        m = maskUtils.decode(rle)
                        mask[m > 0] = 1
        
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
        cv2.imwrite(output_path, mask * 255)
    print(f"Saved binary masks` to {output_path}")

def read_image_files(image_path):
    """read image files individual and from a directory
    """
    
    if os.path.isdir(image_path):
        return [os.path.join(image_path, file_name) for file_name in os.listdir(image_path)]
    else:
        return [image_path]

