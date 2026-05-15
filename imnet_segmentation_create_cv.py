import os
import torch
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse

# Paths - EDIT THESE TO YOUR SETUP
IMNET_ROOT = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/ILSVRC2012"  # your root
ANNOT_DIR = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/imnet_bbox/Annotation/Annotation"  # from step 3
CLASSES_FILE = "/home/fr/fr_fr/fr_ad457/procedural/data/imagenet100.txt"  # your synset list
OUTPUT_DIR = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/imnet_test_subset_2/masks"  # will create masks/<synset>/<img>.png
SAM_CKPT = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/checkpoints/sam_vit_h.pth"
device = "cuda"

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(device)
predictor = SamPredictor(sam)

def get_imagenet_bbox(annot_dir, synset, img_base):
    xml_path = os.path.join(annot_dir, synset, f"{img_base}.xml")
    # print(f"Looking for XML bbox at {xml_path}...")
    if not os.path.exists(xml_path):
        # print(f"  No XML found for {img_base} in {synset}")
        return None

    tree = ET.parse(xml_path)
    bbox = tree.find(".//bndbox")
    if bbox is None:
        return None

    x1 = int(bbox.find("xmin").text)
    y1 = int(bbox.find("ymin").text)
    x2 = int(bbox.find("xmax").text)
    y2 = int(bbox.find("ymax").text)
    return [x1, y1, x2, y2]

def get_auto_box(image_path):
    """
    Return bbox as [x1, y1, x2, y2] for SAM.
    Fallback order:
    1) Largest contour bbox
    2) Largest connected non-black region
    3) Center box
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur a bit to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Try both foreground assumptions: normal and inverted
    candidates = []
    for binary in [thresh, cv2.bitwise_not(thresh)]:
        # Clean small noise
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.01 * h * w:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            candidates.append((area, [x, y, x + bw, y + bh]))

    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    # Fallback: non-black pixels
    mask = gray > 10
    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return [int(x1), int(y1), int(x2), int(y2)]

    # Final fallback: centered box
    return [w // 4, h // 4, 3 * w // 4, 3 * h // 4]


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

def segmentation_singe_image(img_file, synset):
    class_out = os.path.join(OUTPUT_DIR, synset)
    os.makedirs(class_out, exist_ok=True)
    class_dir = os.path.join(IMNET_ROOT, "train", synset)
    img_path = os.path.join(class_dir, img_file)
    # print("Masking", img_path)
    img_base = img_file.replace('.JPEG', '')
    # print("imgbase", img_base)
    
    # Try official ImageNet XML bbox first
    bbox = get_imagenet_bbox(ANNOT_DIR, synset, img_base)
    if bbox is None:
        # Fallback: OpenCV-based auto box
        # bbox = get_auto_box(img_path)
        # if bbox is None:
        # print(f"No XML bbox for {img_file}, skipping")
        return False

    # # overlay bbox on image and save for debugging
    # debug_img = cv2.imread(img_path)
    # cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    # debug_path = os.path.join(class_out, f"{img_base}_bbox.jpg")
    # cv2.imwrite(debug_path, debug_img)
    
    # Generate mask
    try:
        mask_pil = generate_mask(img_path, bbox, predictor)
        mask_path = os.path.join(class_out, f"{img_base}.png")
        mask_pil.save(mask_path)
        # print(f"Processed {img_file} with XML bbox - mask saved to {mask_path}")
        # processed += 1
    except Exception as e:
        print(f"Error {img_file}: {e}")
    return True

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read 100 classes
    with open(CLASSES_FILE) as f:
        classes = [line.strip() for line in f.readlines()][:2]
    
    total_images = 0
    processed = 0
    missing_xml = 0
    
    for synset in tqdm(classes, desc="Classes"):
        class_dir = os.path.join(IMNET_ROOT, "val", synset)
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
                missing_xml += 1
                # Fallback: OpenCV-based auto box
                bbox = get_auto_box(img_path)
                if bbox is None:
                    print(f"No box for {img_file}")
                    continue

            # overlay bbox on image and save for debugging
            debug_img = cv2.imread(img_path)
            cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            debug_path = os.path.join(class_out, f"{img_base}_bbox.jpg")
            cv2.imwrite(debug_path, debug_img)
            
            # Generate mask
            try:
                mask_pil = generate_mask(img_path, bbox, predictor)
                mask_path = os.path.join(class_out, f"{img_base}.png")
                mask_pil.save(mask_path)
                processed += 1
            except Exception as e:
                print(f"Error {img_file}: {e}")
    
    print(f"Done! Processed {processed}/{total_images} images, {missing_xml} used auto-box")

if __name__ == "__main__":
    main()
