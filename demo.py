from damage_detector.detector import Detector
if __name__ == '__main__':
    print("Enter image path: ")
    image_path = input()
    Detector.detect_scratches(image_path)
