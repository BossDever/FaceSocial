import sys
import os
import unittest
import numpy as np
import cv2

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.face_detector import FaceDetector

class TestFaceDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector()
        
        # Create a simple test image (blank with a face-like circle)
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(self.test_image, (150, 150), 50, (255, 255, 255), -1)
        cv2.circle(self.test_image, (130, 130), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(self.test_image, (170, 130), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(self.test_image, (150, 170), (20, 10), 0, 0, 180, (0, 0, 0), -1)  # Mouth
    
    def test_detector_initialization(self):
        """Test if the detector initializes correctly"""
        self.assertIsNotNone(self.detector.detector)
    
    def test_extract_face(self):
        """Test if extract_face method works"""
        bbox = [100, 100, 100, 100]  # [x, y, width, height]
        face_img = self.detector.extract(self.test_image, bbox)
        self.assertEqual(face_img.shape[:2], (160, 160))

if __name__ == '__main__':
    unittest.main()