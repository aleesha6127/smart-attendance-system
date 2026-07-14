"""
LBPH Face Recognition Module - Using OpenCV's Built-in Face Recognizer
This is a proven, robust algorithm specifically designed for face recognition.
"""
import cv2
import numpy as np
import os
import pickle

class LBPHFaceRecognizer:
    def __init__(self, model_path='models/lbph_model.yml', labels_path='models/lbph_labels.pkl'):
        self.model_path = model_path
        self.labels_path = labels_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create LBPH face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=70    # Stricter threshold to prevent misidentification
        )
        
        # Label mapping: id -> name
        self.label_to_name = {}
        self.name_to_label = {}
        self.is_trained = False
        
        # Load existing model if available
        self.load_model()
    
    def load_model(self):
        """Load trained model and labels"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
                self.recognizer.read(self.model_path)
                with open(self.labels_path, 'rb') as f:
                    data = pickle.load(f)
                    self.label_to_name = data['label_to_name']
                    self.name_to_label = data['name_to_label']
                self.is_trained = True
                print(f"✓ LBPH Model loaded. {len(self.label_to_name)} users registered.")
                return True
        except Exception as e:
            print(f"Could not load LBPH model: {e}")
        return False
    
    def save_model(self):
        """Save trained model and labels"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.recognizer.write(self.model_path)
        with open(self.labels_path, 'wb') as f:
            pickle.dump({
                'label_to_name': self.label_to_name,
                'name_to_label': self.name_to_label
            }, f)
        print(f"✓ LBPH Model saved.")
    
    def train_from_dataset(self, dataset_path='dataset'):
        """Train the recognizer from the dataset folder"""
        faces = []
        labels = []
        current_label = 0
        
        self.label_to_name = {}
        self.name_to_label = {}
        
        print(f"Training LBPH from dataset: {dataset_path}")
        
        # Process students
        students_path = os.path.join(dataset_path, 'students')
        if os.path.exists(students_path):
            for user_folder in os.listdir(students_path):
                user_path = os.path.join(students_path, user_folder)
                if os.path.isdir(user_path):
                    user_id = user_folder
                    
                    # Assign label
                    if user_id not in self.name_to_label:
                        self.name_to_label[user_id] = current_label
                        self.label_to_name[current_label] = user_id
                        current_label += 1
                    
                    label = self.name_to_label[user_id]
                    
                    # Process each image
                    for img_file in os.listdir(user_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(user_path, img_file)
                            image = cv2.imread(img_path)
                            if image is None:
                                continue
                            
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            
                            # Apply CLAHE to improve training quality (must match recognition preprocessing)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            gray = clahe.apply(gray)
                            
                            # Detect face in registration image
                            detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                            
                            if len(detected_faces) > 0:
                                x, y, w, h = detected_faces[0]
                                face_roi = gray[y:y+h, x:x+w]
                                
                                # TIGHT CROP: Use the face_roi directly without expansion
                                # This excludes hair/hijab/background
                                face_resized = cv2.resize(face_roi, (200, 200))
                                
                                # Add original
                                faces.append(face_resized)
                                labels.append(label)
                                
                                # Add augmentations for robustness (Crucial since we only have 1 image per student!)
                                # 1. Flipped
                                faces.append(cv2.flip(face_resized, 1))
                                labels.append(label)
                                
                                # 2. Slightly brighter
                                bright = cv2.convertScaleAbs(face_resized, alpha=1.2, beta=15)
                                faces.append(bright)
                                labels.append(label)
                                
                                # 3. Slightly darker
                                dark = cv2.convertScaleAbs(face_resized, alpha=0.8, beta=-15)
                                faces.append(dark)
                                labels.append(label)

                                # 4. Gaussian Blur (Simulate out-of-focus webcam)
                                blurred = cv2.GaussianBlur(face_resized, (5, 5), 0)
                                faces.append(blurred)
                                labels.append(label)

                                # 5. Noise (Simulate webcam grain)
                                noise = np.zeros(face_resized.shape, np.uint8)
                                cv2.randn(noise, 0, 10)  # low noise
                                noisy_face = cv2.add(face_resized, noise)
                                faces.append(noisy_face)
                                labels.append(label)

                                # 6. Zoom/Scale (Simulate distance changes)
                                # Crop center and resize back
                                h, w = face_resized.shape
                                center_x, center_y = w // 2, h // 2
                                crop_size = int(h * 0.9)  # 90% crop
                                x1 = center_x - crop_size // 2
                                y1 = center_y - crop_size // 2
                                zoomed = cv2.resize(face_resized[y1:y1+crop_size, x1:x1+crop_size], (200, 200))
                                faces.append(zoomed)
                                labels.append(label)
                                
                                print(f"  Added {user_id} (7 samples - Augmented)")
                            else:
                                # Use whole image as face (fallback)
                                face_resized = cv2.resize(gray, (200, 200))
                                faces.append(face_resized)
                                labels.append(label)
                                print(f"  Added {user_id} (1 sample, no face detected)")
        
        # Process teachers
        teachers_path = os.path.join(dataset_path, 'teachers')
        if os.path.exists(teachers_path):
            for user_folder in os.listdir(teachers_path):
                user_path = os.path.join(teachers_path, user_folder)
                if os.path.isdir(user_path):
                    user_id = user_folder
                    
                    if user_id not in self.name_to_label:
                        self.name_to_label[user_id] = current_label
                        self.label_to_name[current_label] = user_id
                        current_label += 1
                    
                    label = self.name_to_label[user_id]
                    
                    for img_file in os.listdir(user_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(user_path, img_file)
                            image = cv2.imread(img_path)
                            if image is None:
                                continue
                            
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            
                            # Apply CLAHE for teachers too
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            gray = clahe.apply(gray)
                            
                            detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                            
                            if len(detected_faces) > 0:
                                x, y, w, h = detected_faces[0]
                                face_roi = gray[y:y+h, x:x+w]
                                # TIGHT CROP to exclude abaya/hijab
                                face_resized = cv2.resize(face_roi, (200, 200))
                                
                                faces.append(face_resized)
                                labels.append(label)
                                faces.append(cv2.flip(face_resized, 1))
                                labels.append(label)
                                
                                print(f"  Added {user_id} (2 samples)")
        
        if len(faces) == 0:
            print("ERROR: No faces found in dataset!")
            return False
        
        # Train the recognizer
        print(f"Training LBPH with {len(faces)} samples from {len(self.label_to_name)} users...")
        self.recognizer.train(faces, np.array(labels))
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        print(f"✓ Training complete! Registered users: {list(self.label_to_name.values())}")
        return True
    
    def recognize_face(self, image):
        """
        Recognize faces in an image.
        Returns: (names, face_locations, confidences)
        """
        if not self.is_trained:
            print("[LBPH] Model not trained!")
            return [], [], []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to handle lighting variations
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        recognized_names = []
        face_locations = []
        confidences = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            # TIGHT CROP for recognition
            face_resized = cv2.resize(face_roi, (200, 200))
            
            # Predict
            label, confidence = self.recognizer.predict(face_resized)
            
            # Lower confidence = better match in LBPH
            # Typical thresholds: <50 = excellent, <80 = good, <100 = possible, >100 = unknown
            print(f"[LBPH] Prediction: label={label}, confidence={confidence:.2f}")
            
            if confidence < 80:  # Stricter threshold: <80 (was 90)
                name = self.label_to_name.get(label, "Unknown")
                print(f"[LBPH] ✓ MATCH: {name} (confidence: {confidence:.2f})")
            else:
                name = "Unknown"
                print(f"[LBPH] ✗ Unknown face (confidence too low: {confidence:.2f})")
            
            recognized_names.append(name)
            face_locations.append((y, x+w, y+h, x))  # (top, right, bottom, left) format
            confidences.append(round(confidence, 1))
        
        return recognized_names, face_locations, confidences
    
    def register_face(self, user_id, image_path):
        """Register a new face - retrains the entire model"""
        # This method exists for compatibility, but for LBPH we recommend
        # saving the image to dataset and calling train_from_dataset()
        
        image = cv2.imread(image_path)
        if image is None:
            return False, "Could not read image"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) == 0:
            return False, "No face detected in image"
        
        # For now, just indicate success - the actual training happens via train_from_dataset
        return True, f"Face detected for {user_id}. Please run training.", None


# Global instance
lbph_recognizer = LBPHFaceRecognizer()


def train_lbph_model():
    """Convenience function to train the LBPH model"""
    return lbph_recognizer.train_from_dataset()


if __name__ == "__main__":
    print("Training LBPH Face Recognizer...")
    train_lbph_model()
