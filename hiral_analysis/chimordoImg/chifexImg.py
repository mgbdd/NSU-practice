from faceAnalyzerImg import face_analyzer
import streamlit as st
import os
import argparse

def get_images(dir_path):
    images = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.webp'):
                full_path = os.path.join(root, file)
                images.append(full_path)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=str, help="path to directory with images")
    args = parser.parse_args()

    images = get_images(args.dir_path)

    try:
        fa = face_analyzer(args.dir_path)
    except RuntimeError as e:
        print(f"Error creating face analyzer object: {e}")


    for index, image_path in enumerate(images, start=1):
        print("\nAnalyzing image ", image_path, f" {index}/{len(images)}")
        fa.analyze_image(image_path)
    
    fa.output_file.close()


if __name__ == "__main__":
    main()
