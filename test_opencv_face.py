import cv2
import sys

print("Testing OpenCV face recognition...")
print("OpenCV version:", cv2.__version__)

try:
    # Test if face module is available
    if hasattr(cv2, 'face'):
        print("✓ cv2.face module is available")

        # Test creating face recognizer
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("✓ LBPH Face Recognizer created successfully")

        # Test cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("✗ Could not load face cascade classifier")
            sys.exit(1)
        else:
            print("✓ Face cascade classifier loaded successfully")

        print("🎉 All face recognition components are working!")

    else:
        print("✗ cv2.face module is not available")
        print("Please install opencv-contrib-python:")
        print("pip install opencv-contrib-python")
        sys.exit(1)

except Exception as e:
    print("✗ Error:", str(e))
    sys.exit(1)