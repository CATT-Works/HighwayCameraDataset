""" Creates a YOLO dataset from a Hugging Face dataset in COCO format.
Author: Przemek Sekula
Created: October 2024
"""

import os
import argparse
from tqdm import tqdm
from datasets import load_dataset


def parse_args():
    """Parse command line arguments
    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Creates an object detection dataset from a "
        "set of annotated images"
    )

    parser.add_argument(
        '-p', '--path',
        type=str,
        default='./datasets/yolo_dataset',
        help='Output folder for the dataset. Default: '
        './datasets/yolo_dataset'
    )
    
    return parser.parse_args()    


def convert_and_save(example, split, path, idx):
    """Converts a single example to YOLO format and saves the image and label
    files.
    Args:
        example (dict): a single example from the dataset
        split (str): 'train' or 'test'
        path (str): output path
        idx (int): index of the example in the split
    """
    if split == 'train':
        image_output_dir = os.path.join(path, 'images', 'train')
        label_output_dir = os.path.join(path, 'labels', 'train')
    elif split == 'test':
        image_output_dir = os.path.join(path, 'images', 'val')
        label_output_dir = os.path.join(path, 'labels', 'val')
    else:
        print(f"Unknown split: {split}")
        return

    # Get the image
    image = example['image']  # PIL Image
    height, width = image.height, image.width
    # Alternatively, you can use:
    # height = example['image_height']
    # width = example['image_width']

    # Get bounding boxes and labels from the 'objects' dictionary
    objects = example['objects']
    # List of [xmin, ymin, width, height] in absolute pixels
    bboxes = objects['bbox']
    labels = objects['category']  # List of label IDs

    # Generate a unique filename for the image
    image_filename = f"{split}_{idx}.jpg"
    image_output_path = os.path.join(image_output_dir, image_filename)

    # Save image
    image.save(image_output_path)

    # Prepare label file content
    label_lines = []
    for bbox, label_id in zip(bboxes, labels):
        # bbox is [xmin, ymin, width, height] in absolute pixel coordinates
        xmin, ymin, width, height = bbox

        # Calculate YOLO format coordinates
        x_center = xmin + width / 2
        y_center = ymin + height / 2

        # Normalize coordinates by image width and height
        x_center_norm = x_center / width
        y_center_norm = y_center / height
        w_norm = width / width
        h_norm = height / height

        # Ensure coordinates are within [0, 1]
        x_center_norm = min(max(x_center_norm, 0.0), 1.0)
        y_center_norm = min(max(y_center_norm, 0.0), 1.0)
        w_norm = min(max(w_norm, 0.0), 1.0)
        h_norm = min(max(h_norm, 0.0), 1.0)

        label_line = f"{label_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}"
        label_lines.append(label_line)

    # Save label file
    label_filename = f"{split}_{idx}.txt"
    label_output_path = os.path.join(label_output_dir, label_filename)
    with open(label_output_path, 'w', encoding='utf-8') as filehandle:
        for line in label_lines:
            filehandle.write(line + '\n')


def main(args):
    """Main function
    Args:
        args (argparse.Namespace): parsed arguments
    """

    dataset_dict = load_dataset('PrzemekS/highway-vehicles')

    # Define class names (ensure this matches your dataset)
    # class_names = ['vehicle', 'truck']  # Update with your actual class names
    # label2id = {label: idx for idx, label in enumerate(class_names)}
    # id2label = {idx: label for idx, label in enumerate(class_names)}

    # Ensure directories exist
    for level1 in ['images', 'labels']:
        for level2 in ['train', 'val']:
            os.makedirs(os.path.join(args.path, level1, level2), exist_ok=True)

    # Process train and test splits
    for split in ['train', 'test']:
        dataset = dataset_dict[split]
        print(f"Processing {split} split...")
        for idx, example in enumerate(tqdm(dataset)):
            convert_and_save(example, split, args.path, idx)

    print("Conversion to YOLO format completed.")


if __name__ == '__main__':
    main(parse_args())
