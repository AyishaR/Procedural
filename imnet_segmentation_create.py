import os
import torch
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, load_image, predict, annotate
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse

# Paths - EDIT THESE TO YOUR SETUP
IMNET_ROOT = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/ILSVRC2012_imnet100"  # your root
ANNOT_DIR = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/imnet_bbox/Annotation/Annotation"  # from step 3
CLASSES_FILE = "/home/fr/fr_fr/fr_ad457/procedural/data/imagenet100.txt"  # your synset list
OUTPUT_DIR = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/imnet_bbox/annotation_masks_n100"  # will create masks/<synset>/<img>.png
SAM_CKPT = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/checkpoints/sam_vit_h.pth"
device = "cpu"

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(device)
predictor = SamPredictor(sam)

count_per_class=100

# Load GroundingDINO (for auto boxes)
gdino_model = load_model(
    "/home/fr/fr_fr/fr_ad457/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
    "/home/fr/fr_fr/fr_ad457/GroundingDINO/groundingdino_swint_ogc.pth")

def get_imagenet_bbox(annot_dir, synset, img_base):
    """Get bbox from official ImageNet XML, return [x,y,w,h] or None"""
    xml_path = os.path.join(annot_dir, synset, f"{synset[:9]}_{img_base}.xml")
    if not os.path.exists(xml_path):
        return None
    
    tree = ET.parse(xml_path)
    bbox = tree.find('.//bndbox')
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = map(int, [bbox.find(tag).text for tag in ['xmin','ymin','xmax','ymax']])
    return [x1, y1, x2-x1, y2-y1]  # xywh format

def get_auto_box(image, class_synset):
    """GroundingDINO auto box proposal for main object"""
    boxes, logits, phrases = predict(
        model=gdino_model, image=image, 
        caption=f"the {class_synset}", 
        box_threshold=0.3,
        text_threshold=0.25
    )
    if len(boxes) == 0:
        return None
    
    # Take highest-confidence box
    best_idx = logits.argmax()
    h, w = image.shape[:2]
    box = boxes[best_idx] * torch.Tensor([w, h, w, h])
    box = box.cpu().numpy().astype(int).tolist()  # xyxy -> xywh
    box[2] -= box[0]
    box[3] -= box[1]
    return box

def generate_mask(image_path, bbox, predictor):
    """SAM mask from bbox prompt, return binary PIL PNG"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(image_rgb)
    input_box = np.array(bbox)
    masks, scores, _ = predictor.predict(
        box=input_box[None, :],
        multimask_output=False  # single best mask
    )
    
    # Binary mask (main object only)
    mask = masks[0] * 255
    mask = mask.astype(np.uint8)
    return Image.fromarray(mask)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read 100 classes
    with open(CLASSES_FILE) as f:
        classes = [line.strip() for line in f.readlines()][:2]
    
    total_images = 0
    processed = 0
    missing_xml = 0
    
    for synset in tqdm(classes, desc="Classes"):
        class_count=0
        class_dir = os.path.join(IMNET_ROOT, "train", synset)
        if not os.path.exists(class_dir):
            print(f"Skipping {synset}: no folder")
            continue
            
        class_out = os.path.join(OUTPUT_DIR, synset)
        os.makedirs(class_out, exist_ok=True)
        
        imgs = [f for f in os.listdir(class_dir) if f.endswith('.JPEG')]
        total_images += len(imgs)
        
        for img_file in tqdm(imgs, desc=synset, leave=False):
            img_path = os.path.join(class_dir, img_file)
            img_base = img_file.replace('.JPEG', '')
            
            # Try official ImageNet XML bbox first
            bbox = get_imagenet_bbox(ANNOT_DIR, synset, img_base)
            if bbox is None:
                continue
                # missing_xml += 1
                # # Fallback: GroundingDINO auto proposal
                # image_source, image = load_image(img_path)
                # bbox = get_auto_box(image, synset)
                # if bbox is None:
                #     print(f"No box for {img_file}")
                #     continue
            
            # Generate mask
            try:
                mask_pil = generate_mask(img_path, bbox, predictor)
                mask_path = os.path.join(class_out, f"{img_base}.png")
                mask_pil.save(mask_path)
                processed += 1
                class_count+=1
            except Exception as e:
                print(f"Error {img_file}: {e}")
            if class_count>=count_per_class:
                print(f"Reached {count_per_class} for {synset}, skipping remaining images.")
                continue
    
    print(f"Done! Processed {processed}/{total_images} images, {missing_xml} used auto-box")

if __name__ == "__main__":
    main()
