import os
import cv2
import argparse
import shutil


def augment_dataset(img_dir, label_dir, output_img_dir, output_label_dir, contrast=1.2, brightness=30):

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image {img_path}")
            continue

        base, ext = os.path.splitext(fname)

        out_orig = os.path.join(output_img_dir, fname)
        cv2.imwrite(out_orig, img)

        aug = cv2.addWeighted(img, contrast, img, 0, brightness)
        aug_name = f"{base}_aug{ext}"
        out_aug = os.path.join(output_img_dir, aug_name)
        cv2.imwrite(out_aug, aug)

        label_file = f"{base}.txt"
        src_label = os.path.join(label_dir, label_file)
        if os.path.exists(src_label):
            dst_orig_label = os.path.join(output_label_dir, label_file)
            dst_aug_label = os.path.join(output_label_dir, f"{base}_aug.txt")
            shutil.copy(src_label, dst_orig_label)
            shutil.copy(src_label, dst_aug_label)
        else:
            print(f"Warning: no label file for image {fname}")


def parse_args():
    parser = argparse.ArgumentParser(description='Augment X-ray dataset for YOLO training')
    parser.add_argument('--input-img', dest='input_img', required=True, help='Path to original images directory')
    parser.add_argument('--input-label', dest='input_label',required=True, help='Path to original labels directory')
    parser.add_argument('--output-img', dest='output_img', required=True, help='Path to save augmented images')
    parser.add_argument('--output-label', dest='output_label', required=True, help='Path to save augmented labels')
    parser.add_argument('--contrast', type=float, default=1.2, help='Contrast factor')
    parser.add_argument('--brightness', type=float, default=30, help='Brightness offset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    augment_dataset(
        img_dir=args.input_img,
        label_dir=args.input_label,
        output_img_dir=args.output_img,
        output_label_dir=args.output_label,
        contrast=args.contrast,
        brightness=args.brightness
    )
    print("Augmentation complete.")
