import cv2
import numpy as np
import os
import pickle
import face_recognition
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Global flag for face_recognition availability
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("[+] face_recognition library available")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[-] face_recognition library not available, using fallback methods")

def get_haarcascade_path():
    """Get the path to Haar cascade files"""
    # Try common locations
    possible_paths = [
        cv2.data.haarcascades,
        'haarcascade_frontalface_default.xml',
        os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
        elif os.path.exists(path) and path.endswith('.xml'):
            return os.path.dirname(path)
    
    # Fallback to current directory
    return ""

class EnhancedFaceRecognition:
    def __init__(self, encodings_path='models/face_encodings.pkl'):
        self.encodings_path = encodings_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.svm_classifier = SVC(C=1.0, kernel='linear', probability=True)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_vector_size = 8100
        self.face_recognition_available = FACE_RECOGNITION_AVAILABLE
        self.load_encodings()
    
    def detect_faces(self, image):
        """
        Detect faces in an image using multiple detection methods for maximum accuracy
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)
        
        # Apply histogram equalization as additional enhancement
        gray_eq = cv2.equalizeHist(gray)
        
        hc_path = get_haarcascade_path()
        face_cascade = cv2.CascadeClassifier(os.path.join(hc_path, 'haarcascade_frontalface_default.xml'))
        if face_cascade.empty():
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Multiple detection attempts with different parameters
        all_faces = []
        
        # Attempt 1: Standard detection
        faces1 = face_cascade.detectMultiScale(gray_clahe, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        all_faces.extend([(x, y, w, h) for (x, y, w, h) in faces1])
        
        # Attempt 2: Relaxed parameters
        faces2 = face_cascade.detectMultiScale(gray_clahe, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25))
        all_faces.extend([(x, y, w, h) for (x, y, w, h) in faces2])
        
        # Attempt 3: More sensitive detection
        faces3 = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.02, minNeighbors=2, minSize=(20, 20))
        all_faces.extend([(x, y, w, h) for (x, y, w, h) in faces3])
        
        # Remove duplicates and keep the best detections
        unique_faces = self._filter_duplicate_faces(all_faces, image.shape)
        
        # Convert to face_recognition format (top, right, bottom, left)
        face_locations = []
        for (x, y, w, h) in unique_faces:
            face_locations.append((y, x+w, y+h, x))
        
        return face_locations
    
    def _filter_duplicate_faces(self, faces, image_shape):
        """
        Filter out duplicate face detections and keep only the best quality ones
        """
        if not faces:
            return []
        
        # Remove exact duplicates
        unique_faces = list(set(faces))
        
        # Sort by area (larger faces first) and remove overlapping detections
        unique_faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        
        filtered_faces = []
        img_h, img_w = image_shape[:2]
        
        for face in unique_faces:
            x, y, w, h = face
            # Check if this face overlaps significantly with already selected faces
            overlap = False
            for selected_face in filtered_faces:
                sx, sy, sw, sh = selected_face
                # Calculate intersection over union (IoU)
                x1 = max(x, sx)
                y1 = max(y, sy)
                x2 = min(x + w, sx + sw)
                y2 = min(y + h, sy + sh)
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = w * h + sw * sh - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    # If faces overlap more than 50%, consider them duplicates
                    if iou > 0.5:
                        overlap = True
                        break
            
            if not overlap:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def assess_face_quality(self, image, face_location):
        """
        Assess the quality of a detected face for registration
        Returns: quality_score (0-100), quality_issues (list of strings)
        """
        top, right, bottom, left = face_location
        face_roi = image[top:bottom, left:right]
        
        if face_roi.size == 0:
            return 0, ["Invalid face region"]
        
        issues = []
        quality_score = 100
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Check brightness
        mean_brightness = np.mean(gray_face)
        if mean_brightness < 40:
            issues.append("Too dark")
            quality_score -= 20
        elif mean_brightness > 215:
            issues.append("Too bright")
            quality_score -= 20
        
        # 2. Check contrast
        contrast = np.std(gray_face)
        if contrast < 25:
            issues.append("Low contrast")
            quality_score -= 15
        
        # 3. Check sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        if laplacian_var < 100:
            issues.append("Blurry image")
            quality_score -= 25
        
        # 4. Check face size (should be adequate)
        face_area = (bottom - top) * (right - left)
        image_area = image.shape[0] * image.shape[1]
        face_ratio = face_area / image_area
        
        if face_ratio < 0.05:  # Less than 5% of image
            issues.append("Face too small")
            quality_score -= 20
        elif face_ratio > 0.5:  # More than 50% of image
            issues.append("Face too large/zoomed")
            quality_score -= 15
        
        # 5. Check for proper face orientation (basic check)
        face_height = bottom - top
        face_width = right - left
        aspect_ratio = face_width / face_height if face_height > 0 else 0
        
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            issues.append("Unusual face orientation")
            quality_score -= 10
        
        # Ensure minimum quality score
        quality_score = max(0, quality_score)
        
        return quality_score, issues
    
    def extract_face_features(self, face_img):
        """
        Extract features from a face image using HOG (Histogram of Oriented Gradients)
        """
        # Resize to standard size for processing
        face_resized = cv2.resize(face_img, (128, 128))
        
        # Convert to grayscale
        if len(face_resized.shape) == 2:
            face_img_gray = face_resized
        else:
            face_img_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
        # Apply CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_img_gray = clahe.apply(face_img_gray)
        
        # Configure HOG Descriptor
        winSize = (128, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        
        # Compute HOG descriptors
        hog_features = hog.compute(face_img_gray).flatten()
        
        # Normalize to unit vector for cosine similarity usage
        norm = np.linalg.norm(hog_features)
        if norm != 0:
            normalized_features = hog_features / norm
        else:
            normalized_features = hog_features
            
        return normalized_features.astype(np.float64)
    
    def encode_face(self, image):
        """
        Encode a face in an image using face_recognition library if available, otherwise fallback
        """
        if self.face_recognition_available:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Try with HOG model first
                face_locations = face_recognition.face_locations(rgb_image, model='hog')
                
                if len(face_locations) == 0:
                    # Try with different number of upsamples
                    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=2)
                
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                if len(face_encodings) > 0:
                    # Find the largest face by area
                    best_face_idx = 0
                    if len(face_locations) > 1:
                        max_area = 0
                        for i, (top, right, bottom, left) in enumerate(face_locations):
                            area = (bottom - top) * (right - left)
                            if area > max_area:
                                max_area = area
                                best_face_idx = i
                    
                    return face_encodings[best_face_idx]
                
                return self._opencv_encode_face(image)
            except Exception as e:
                print(f"[encode_face] Error: {e}, falling back to OpenCV")
                return self._opencv_encode_face(image)
        else:
            return self._opencv_encode_face(image)
    
    def _opencv_encode_face(self, image):
        """
        Internal method for OpenCV-based face encoding
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hc_path = get_haarcascade_path()
        face_cascade = cv2.CascadeClassifier(os.path.join(hc_path, 'haarcascade_frontalface_default.xml'))
        if face_cascade.empty():
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Pick the largest face detected
            best_face_idx = 0
            if len(faces) > 1:
                max_area = 0
                for i, (x, y, w, h) in enumerate(faces):
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_face_idx = i
            
            x, y, w, h = faces[best_face_idx]
            face_roi = image[y:y+h, x:x+w]
            face_features = self.extract_face_features(face_roi)
            return face_features
        return None
    
    def _select_best_face(self, image, face_locations):
        """
        Select the best face from multiple detections based on size and quality
        """
        if len(face_locations) == 1:
            return 0
        
        best_idx = 0
        best_score = 0
        
        for i, location in enumerate(face_locations):
            top, right, bottom, left = location
            face_area = (bottom - top) * (right - left)
            
            # Quality assessment for this face
            quality_score, _ = self.assess_face_quality(image, location)
            
            # Combined score: 70% area, 30% quality
            combined_score = (face_area * 0.7) + (quality_score * 30)
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = i
        
        return best_idx
    
    def _extract_face_encoding(self, rgb_image, face_location, isolated_face_image):
        """
        Extract face encoding with multiple fallback methods
        """
        face_encoding = None
        
        # Method 1: Direct encoding using face_recognition
        if self.face_recognition_available:
            try:
                face_encodings = face_recognition.face_encodings(rgb_image, [face_location])
                if face_encodings:
                    face_encoding = face_encodings[0]
            except Exception as e:
                print(f"[Encoding] face_recognition failed: {e}")
                self.face_recognition_available = False
        
        # Method 2: Fallback encoding using isolated face
        if face_encoding is None:
            try:
                face_encoding = self.encode_face(isolated_face_image)
            except Exception as e:
                print(f"[Encoding] Fallback method failed: {e}")
        
        return face_encoding
    
    def _generate_face_variations(self, primary_encoding, face_roi):
        """
        Generate multiple face encoding variations for robust recognition
        """
        variations = [primary_encoding]
        
        if not self.face_recognition_available and face_roi is not None:
            try:
                h, w = face_roi.shape[:2]
                center = (w // 2, h // 2)
                
                # Variation 1: Horizontal flip
                flip_roi = cv2.flip(face_roi, 1)
                variations.append(self.extract_face_features(flip_roi))
                
                # Variation 2: Brighter
                bright_roi = cv2.convertScaleAbs(face_roi, alpha=1.15, beta=20)
                variations.append(self.extract_face_features(bright_roi))
                
                # Variation 3: Darker
                dark_roi = cv2.convertScaleAbs(face_roi, alpha=0.85, beta=-15)
                variations.append(self.extract_face_features(dark_roi))
                
                # Variation 4: Slight rotation (+3 degrees)
                M_plus = cv2.getRotationMatrix2D(center, 3, 1.0)
                rot_plus = cv2.warpAffine(face_roi, M_plus, (w, h))
                variations.append(self.extract_face_features(rot_plus))
                
                # Variation 5: Slight rotation (-3 degrees)
                M_minus = cv2.getRotationMatrix2D(center, -3, 1.0)
                rot_minus = cv2.warpAffine(face_roi, M_minus, (w, h))
                variations.append(self.extract_face_features(rot_minus))
                
                # Variation 6: Blur simulation
                blurred_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
                variations.append(self.extract_face_features(blurred_roi))
                
            except Exception as e:
                print(f"[Variations] Error generating variations: {e}")
        
        # Filter out None values
        variations = [v for v in variations if v is not None]
        
        return variations
    
    def _check_for_duplicates(self, user_id, new_encodings):
        """
        Enhanced duplicate checking with multiple verification methods
        Returns: {'is_unique': bool, 'existing_user': str, 'confidence': float}
        """
        result = {'is_unique': True, 'existing_user': None, 'confidence': 0.0}
        
        if not new_encodings:
            return result
        
        # Prepare valid known encodings
        valid_known_encodings = []
        valid_known_names = []
        
        for i, known_encoding in enumerate(self.known_face_encodings):
            if (known_encoding is not None and 
                hasattr(known_encoding, 'shape') and 
                known_encoding.shape == new_encodings[0].shape):
                valid_known_encodings.append(known_encoding)
                valid_known_names.append(self.known_face_names[i])
        
        if not valid_known_encodings:
            return result
        
        # Method 1: face_recognition distance checking
        if self.face_recognition_available:
            try:
                max_confidence = 0.0
                best_match_user = None
                
                for new_enc in new_encodings:
                    face_distances = face_recognition.face_distance(valid_known_encodings, new_enc)
                    if len(face_distances) > 0:
                        closest_idx = np.argmin(face_distances)
                        distance = face_distances[closest_idx]
                        # Convert distance to confidence (lower distance = higher confidence)
                        confidence = max(0, (1.0 - distance) * 100)
                        
                        if confidence > max_confidence:
                            max_confidence = confidence
                            best_match_user = valid_known_names[closest_idx]
                
                # Strict threshold: 75% confidence required to consider duplicate
                if max_confidence > 75.0 and best_match_user != user_id:
                    result['is_unique'] = False
                    result['existing_user'] = best_match_user
                    result['confidence'] = max_confidence
                    return result
                    
            except Exception as e:
                print(f"[Duplicate Check] face_recognition method failed: {e}")
                self.face_recognition_available = False
        
        # Method 2: Cosine similarity fallback
        max_similarity = 0.0
        best_match_user = None
        
        for new_enc in new_encodings:
            for i, known_enc in enumerate(valid_known_encodings):
                try:
                    # Cosine similarity
                    dot_product = np.dot(new_enc, known_enc)
                    norm_product = np.linalg.norm(new_enc) * np.linalg.norm(known_enc)
                    if norm_product != 0:
                        similarity = dot_product / norm_product
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match_user = valid_known_names[i]
                except Exception as e:
                    continue
        
        # Convert similarity to percentage and check threshold
        similarity_percent = max_similarity * 100
        # Strict threshold: 85% similarity required to consider duplicate
        if similarity_percent > 85.0 and best_match_user != user_id:
            result['is_unique'] = False
            result['existing_user'] = best_match_user
            result['confidence'] = similarity_percent
            return result
        
        return result
    
    def _get_user_role(self, user_id):
        """
        Determine user role from user_id
        """
        if user_id.startswith('student'):
            return 'student'
        elif user_id.startswith('teacher'):
            return 'teacher'
        else:
            return 'user'
    
    def register_face(self, user_id, image_path, bypass_duplicate_check=False):
        """
        Register a face for a user with enhanced accuracy and duplicate checking
        Returns: (success, message, extra_info)
        """
        if not os.path.exists(image_path):
            return False, "Image file not found", None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return False, "Could not read image file", None
        
        # Convert to RGB for face_recognition library
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces with enhanced method
        face_locations = self.detect_faces(image)
        
        if not face_locations:
            return False, "No face detected in image. Please ensure face is clearly visible.", None
        
        # Select the best face (largest and highest quality)
        best_face_idx = self._select_best_face(image, face_locations)
        selected_location = face_locations[best_face_idx]
        
        # Assess face quality before proceeding
        quality_score, quality_issues = self.assess_face_quality(image, selected_location)
        
        if quality_score < 60:  # Minimum quality threshold
            issues_str = ", ".join(quality_issues) if quality_issues else "Poor image quality"
            return False, f"Face image quality too low ({quality_score}%). Issues: {issues_str}. Please retake with better lighting and clarity.", None
        
        # Crop image to the isolated face with margin
        top, right, bottom, left = selected_location
        h, w, _ = image.shape
        margin_h = int((bottom - top) * 0.25)  # Increased margin
        margin_w = int((right - left) * 0.25)
        
        new_top = max(0, top - margin_h)
        new_bottom = min(h, bottom + margin_h)
        new_left = max(0, left - margin_w)
        new_right = min(w, right + margin_w)
        
        isolated_face_image = image[new_top:new_bottom, new_left:new_right]
        
        # Extract face encoding with enhanced method
        face_encoding = self._extract_face_encoding(rgb_image, selected_location, isolated_face_image)
        
        if face_encoding is None:
            return False, "Could not extract facial features. Please ensure face is properly positioned.", None
        
        # Reload encodings if needed
        if len(self.known_face_encodings) != len(self.known_face_names):
            self.load_encodings()

        # Remove previous data for this user to keep it clean
        current_indices = [i for i, name in enumerate(self.known_face_names) if name == user_id]
        for index in sorted(current_indices, reverse=True):
            self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)
        
        # Generate robust face encodings with multiple variations
        new_encodings = self._generate_face_variations(face_encoding, isolated_face_image)
        
        print(f"[Register Face] Generated {len(new_encodings)} variations for robust recognition")
        
        # Enhanced duplicate checking with multiple verification methods
        if not bypass_duplicate_check:
            duplicate_result = self._check_for_duplicates(user_id, new_encodings)
            if not duplicate_result['is_unique']:
                existing_user = duplicate_result['existing_user']
                existing_role = self._get_user_role(existing_user)
                confidence = duplicate_result['confidence']
                
                print(f"[Duplicate Blocked] Match found: {existing_user} (Confidence: {confidence:.2f})")
                return False, f"Face already registered to {existing_role} '{existing_user}' (confidence: {confidence:.1f}%). Each profile must have a unique face.", existing_user
        
        # Add new face data
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(user_id)
        
        # Save encodings
        self.save_encodings()
        
        print(f"[+] Successfully registered {user_id} with quality score: {quality_score}%")
        return True, f"Face registered successfully for {user_id} (Quality: {quality_score}%)", None
    
    def recognize_face(self, image):
        """
        Recognize faces in an image
        Returns: (names, locations, confidences)
        """
        # Implementation would go here
        pass
    
    def load_encodings(self):
        """Load face encodings from file"""
        if os.path.exists(self.encodings_path):
            try:
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                    print(f"Loaded {len(self.known_face_encodings)} face encodings")
            except Exception as e:
                print(f"Error loading encodings: {e}")
                self.known_face_encodings = []
                self.known_face_names = []
        else:
            self.known_face_encodings = []
            self.known_face_names = []
    
    def save_encodings(self):
        """Save face encodings to file"""
        try:
            os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
            with open(self.encodings_path, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            print(f"Saved {len(self.known_face_encodings)} face encodings")
        except Exception as e:
            print(f"Error saving encodings: {e}")

# Global instance
face_recognizer = EnhancedFaceRecognition()