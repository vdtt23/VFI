import cv2
import sys

def check_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"Cannot open: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frames / fps if fps > 0 else 0

    print(f"\nVideo: {path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {frames}")
    print(f"Duration: {duration:.2f} seconds")

    cap.release()


check_video(sys.argv[1])
