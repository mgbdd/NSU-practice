from video_classification.faceAnalyzer import face_analyzer
import argparse
import os

def get_video_files(dir_path):
    video_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.mp4'):
                full_path = os.path.join(root, file)
                video_files.append(full_path)
    return video_files

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=str, help="path to directory with videos")
    args = parser.parse_args()

    fa = face_analyzer()
    video_files = get_video_files(args.dir_path)
    
    for index, video_path in enumerate(video_files, start=1):
        print("Analyzing ", video_path, f" {index}/{len(video_files)}")
        fa.analyze_video(video_path)


if __name__ == "__main__":
    main()


