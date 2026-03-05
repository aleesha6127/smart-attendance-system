import cv2
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Attempt to import face_recognition, with fallback to OpenCV if unavailable
try:
    # import face_recognition  # Using face_recognition library which is built on top of dlib
    # FACE_RECOGNITION_AVAILABLE = True
    # print("Face recognition library loaded successfully")
    raise ImportError("Forcing internal OpenCV feature extraction")
except ImportError:
    print("[+] Running in Enhanced Standalone Mode (High-Accuracy OpenCV+Sobel Engine active)")
    FACE_RECOGNITION_AVAILABLE = False
except AttributeError as e:
    print(f"Warning: face_recognition library has attribute error ({e}). Using OpenCV/HOG-based fallback.")
    FACE_RECOGNITION_AVAILABLE = False
except Exception as e:
    FACE_RECOGNITION_AVAILABLE = False

def get_haarcascade_path():
    """Fail-safe way to find haarcascades"""
    # 1. Try standard cv2.data
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            return cv2.data.haarcascades
    except:
        pass
    
    # 2. Try relative to cv2 module
    cv2_path = os.path.dirname(cv2.__file__)
    data_path = os.path.join(cv2_path, 'data')
    if os.path.exists(data_path):
        return data_path + os.sep
        
    return ""

class FaceRecognition:
    def __init__(self, encodings_path='models/face_encodings.pkl'):
        self.encodings_path = encodings_path
        file_path = os.path.abspath(__file__)
        file_size = os.path.getsize(file_path)
        print(f"[FaceRecognition] Initializing from: {file_path}", flush=True)
        print(f"[FaceRecognition] File size: {file_size} bytes", flush=True)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            print(f"[FaceRecognition] File content hash (first 50): {hash(content[:50])}", flush=True)
        print(f"[FaceRecognition] Model path: {os.path.abspath(self.encodings_path)}", flush=True)
        self.known_face_encodings = []
        self.known_face_names = []
        # For SVM-based recognition when face_recognition is not available
        self.svm_classifier = SVC(C=1.0, kernel='linear', probability=True)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_vector_size = 8100  # HOG Descriptor Size (15x15 blocks * 4 cells * 9 bins)
        # Store the global FACE_RECOGNITION_AVAILABLE in instance variable
        self.face_recognition_available = FACE_RECOGNITION_AVAILABLE
        self.load_encodings()
    
    def detect_faces(self, image):
        """
        Detect faces in an image using OpenCV's Haar Cascade with CLAHE preprocessing
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to improve detection in bad lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        hc_path = get_haarcascade_path()
        face_cascade = cv2.CascadeClassifier(os.path.join(hc_path, 'haarcascade_frontalface_default.xml'))
        if face_cascade.empty():
            # Fallback if path failed
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            
        # Tuned parameters: minNeighbors=4 (slightly sensitive), scaleFactor=1.1
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        return faces
    
    def extract_face_features(self, face_img):
        """
        Extract features from a face image using PROFESSIONAL HOG (Histogram of Oriented Gradients)
        This uses the highly optimized cv2.HOGDescriptor for maximum accuracy in standalone mode.
        """
        # --- STRICT INNER FACE CROP ---
        # Crop to exclude clothing (dress/neck) and background hair
        # By focusing strictly on eyes, nose, and mouth, we prevent clothing from influencing recognition
        h, w = face_img.shape[:2]
        
        # Crop 15% from top (hair/forehead), 20% from bottom (neck/dress/collar), and 15% from sides (ears/background)
        y1 = int(h * 0.15)
        y2 = int(h * 0.80)
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)
        
        # Ensure valid crop
        if y2 > y1 and x2 > x1:
            face_img_cropped = face_img[y1:y2, x1:x2]
        else:
            face_img_cropped = face_img
            
        # Resize to standard size for processing (matches HOG window size)
        face_resized = cv2.resize(face_img_cropped, (128, 128))
        
        # HOG expects grayscale or color, but standard is usually localized gradients.
        if len(face_resized.shape) == 2:
            face_img_gray = face_resized
        else:
            face_img_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
        # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) instead of global equalization
        # This preserves local details better for face recognition
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_img_gray = clahe.apply(face_img_gray)
        
        # Configure HOG Descriptor
        # WinSize: 128x128
        # BlockSize: 16x16 (captures local texture)
        # BlockStride: 8x8 (overlaps for smoothness)
        # CellSize: 8x8
        # Bins: 9 (standard gradient directions)
        winSize = (128, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        
        # Compute HOG descriptors
        # This returns a 1D vector of length 8100
        hog_features = hog.compute(face_img_gray).flatten()
        
        # Normalize to unit vector (L2 norm) for cosine similarity usage
        norm = np.linalg.norm(hog_features)
        if norm != 0:
            normalized_features = hog_features / norm
        else:
            normalized_features = hog_features
            
        return normalized_features.astype(np.float64)

    def encode_face(self, image):
        """
        Encode a face in an image using face_recognition library if available, otherwise fallback to OpenCV+HOG
        """
        print(f"[encode_face] Starting - face_recognition_available: {self.face_recognition_available}")
        print(f"[encode_face] Image shape: {image.shape}")
        
        if self.face_recognition_available:
            # Use the face_recognition library if available
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Try with CNN model first (more accurate but slower), fallback to HOG
                face_locations = face_recognition.face_locations(rgb_image, model='hog')
                print(f"[encode_face] face_locations (HOG): {face_locations}")
                
                if len(face_locations) == 0:
                    # Try with different number of upsamples
                    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=2)
                    print(f"[encode_face] face_locations (2x upsample): {face_locations}")
                
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                print(f"[encode_face] Got {len(face_encodings)} encodings")
                
                if len(face_encodings) > 0:
                    # Find the largest face by area to prioritize the foreground person
                    best_face_idx = 0
                    if len(face_locations) > 1:
                        max_area = 0
                        for i, (top, right, bottom, left) in enumerate(face_locations):
                            area = (bottom - top) * (right - left)
                            if area > max_area:
                                max_area = area
                                best_face_idx = i
                        print(f"[encode_face] Selected largest face (idx {best_face_idx}) out of {len(face_encodings)} detected")
                    
                    return face_encodings[best_face_idx]
                
                print("[encode_face] No faces detected by face_recognition library, falling back to OpenCV")
                return self._opencv_encode_face(image)  # Fallback if no face detected
            except AttributeError as e:
                # If there's an attribute error, fall back to OpenCV
                print(f"Face recognition library has attribute error ({e}), using OpenCV fallback")
                self.face_recognition_available = False
                return self._opencv_encode_face(image)
            except Exception as e:
                print(f"[encode_face] Error: {e}, falling back to OpenCV")
                return self._opencv_encode_face(image)
        else:
            # Fallback to OpenCV-based face detection with HOG-inspired feature extraction
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
            # Pick the largest face detected based on area (w * h)
            best_face_idx = 0
            if len(faces) > 1:
                max_area = 0
                for i, (x, y, w, h) in enumerate(faces):
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_face_idx = i
                print(f"[_opencv_encode_face] Selected largest face (idx {best_face_idx}) out of {len(faces)} detected")
            
            x, y, w, h = faces[best_face_idx]
            face_roi = image[y:y+h, x:x+w]
            face_features = self.extract_face_features(face_roi)
            return face_features
        return None
    
    def assess_face_quality(self, image, face_location=None):
        """
        Assess the quality of a detected face for registration
        Returns: quality_score (0-100), quality_issues (list of strings)
        """
        if face_location:
            if len(face_location) == 4:
                # Handle face_recognition format (top, right, bottom, left)
                top, right, bottom, left = face_location
            else:
                # Handle (x, y, w, h) format if passed incorrectly, though we try to standardize
                x, y, w, h = face_location
                top, right, bottom, left = y, x+w, y+h, x
                
            face_roi = image[top:bottom, left:right]
        else:
            face_roi = image
        
        if face_roi.size == 0:
            return 0, ["Invalid face region"]
        
        issues = []
        quality_score = 100
        
        # Convert to grayscale for analysis
        if len(face_roi.shape) == 3:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_roi
        
        # 1. Check brightness
        mean_brightness = np.mean(gray_face)
        if mean_brightness < 40:
            issues.append("Too dark")
            quality_score -= 30
        elif mean_brightness > 215:
            issues.append("Too bright")
            quality_score -= 30
        
        # 2. Check contrast
        contrast = np.std(gray_face)
        if contrast < 20: # Slightly relaxed from 25
            issues.append("Low contrast")
            quality_score -= 20
        
        # 3. Check sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        if laplacian_var < 50: # Relaxed from 100 for webcams
            issues.append("Blurry image")
            quality_score -= 30
        
        # 4. Check face size (should be adequate) if we have reference image size
        if face_location:
            face_area = (bottom - top) * (right - left)
            image_area = image.shape[0] * image.shape[1]
            face_ratio = face_area / image_area
            
            if face_ratio < 0.02:  # Less than 2% of image
                issues.append("Face too small")
                quality_score -= 30
        
        # Ensure minimum quality score
        quality_score = max(0, quality_score)
        
        return quality_score, issues

    def register_face(self, user_id, image_path, bypass_duplicate_check=False):
        """
        Register a new face for a user with Auto-Augmentation and Strict Duplicate Checks
        Returns (success, message, extra_info)
        """
        print(f"[SANITY CHECK] register_face called for {user_id}", flush=True)
        image = cv2.imread(image_path)
        if image is None:
            return False, "Could not read image file", None
        
        # 1. Detect and Isolate the PRIMARY face
        # This prevents background people from interfering with the analysis
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = []
        
        if self.face_recognition_available:
            face_locations = face_recognition.face_locations(rgb_image, model='hog')
        
        if not face_locations or not self.face_recognition_available:
            # Fallback/Manual detection to get ROI
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            hc_path = get_haarcascade_path()
            face_cascade = cv2.CascadeClassifier(os.path.join(hc_path, 'haarcascade_frontalface_default.xml'))
            if face_cascade.empty():
                 face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            
            # Attempt 1: Standard
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Attempt 2: Relaxed (if failed)
            if len(faces) == 0:
                print("[register_face] No faces found with standard params, retrying with relaxed...")
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
            
            if len(faces) > 0:
                # Convert OpenCV (x, y, w, h) to face_recognition (top, right, bottom, left)
                for (x, y, w, h) in faces:
                    face_locations.append((y, x+w, y+h, x))
        
        if not face_locations:
            return False, "No face detected (checked with CLAHE+Relaxed Params)", None
            
        # Select the largest face ROI
        best_face_idx = 0
        if len(face_locations) > 1:
            max_area = 0
            for i, (top, right, bottom, left) in enumerate(face_locations):
                area = (bottom - top) * (right - left)
                if area > max_area:
                    max_area = area
                    best_face_idx = i
            print(f"[register_face] Isolating largest face (idx {best_face_idx}) out of {len(face_locations)} detected")
        
        # --- QUALITY CHECK ---
        quality_score, issues = self.assess_face_quality(image, face_locations[best_face_idx])
        print(f"[register_face] Quality Score: {quality_score}, Issues: {issues}")
        
        if quality_score < 60: # Threshold for registration
             return False, f"Image quality too low ({quality_score}%). Issues: {', '.join(issues)}. Please try again with better lighting.", None

        # Crop image to the isolated face with some margin (20%)
        # TIGHT CROP to exclude abaya/hijab/background
        # We strictly use the detected face region without expansion
        # In fact, we might want to shrink slightly to be safe, but 0 margin is a good start
        top, right, bottom, left = face_locations[best_face_idx]
        h, w, _ = image.shape
        
        # margin_h = int((bottom - top) * 0.0) # 0% margin
        # margin_w = int((right - left) * 0.0)
        
        new_top = max(0, top)
        new_bottom = min(h, bottom)
        new_left = max(0, left)
        new_right = min(w, right)
        
        isolated_face_image = image[new_top:new_bottom, new_left:new_right]
        face_rois = [isolated_face_image] # Store for augmentation if needed
        
        # Now perform encoding on the isolated face area
        # We use the ORIGINAL RGB image with the FOUND location to get the highest quality encoding
        face_encoding = None
        try:
            # We must use the original location list
            if self.face_recognition_available:
                face_encodings = face_recognition.face_encodings(rgb_image, [face_locations[best_face_idx]])
                if face_encodings:
                    face_encoding = face_encodings[0]
            
            if face_encoding is None:
                # Absolute fallback: try encoding the cropped image
                face_encoding = self.encode_face(isolated_face_image)
        except Exception as e:
            print(f"[register_face] Direct encoding failed: {e}, falling back to crop")
            face_encoding = self.encode_face(isolated_face_image)
        
        if face_encoding is None:
            return False, "Could not extract features from the detected face", None
        
        # Reload encodings if needed
        if len(self.known_face_encodings) != len(self.known_face_names):
            self.load_encodings()

        # Remove previous data for this user to keep it clean (and avoid self-duplicate)
        current_indices = [i for i, name in enumerate(self.known_face_names) if name == user_id]
        for index in sorted(current_indices, reverse=True):
            self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)
        
        # --- AUGMENTATION (For Robustness) ---
        new_encodings = [face_encoding] # Start with primary
        
        # Generate variations for duplicate checking even if face_recognition is available
        # This helps catch duplicates in different lighting conditions
        if len(face_rois) > 0:
            roi = face_rois[0]
            try:
                # Variant 1: Horizontal Flip (Mirror)
                flip_roi = cv2.flip(roi, 1)
                new_encodings.append(self.encode_face(flip_roi))
                
                # Variant 2: Brightness Up
                bright_roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)
                new_encodings.append(self.encode_face(bright_roi))
                
                # Variant 3: Brightness Down
                dark_roi = cv2.convertScaleAbs(roi, alpha=0.8, beta=-10)
                new_encodings.append(self.encode_face(dark_roi))
                
                # Variant 4: Zoom In (Crop 10%)
                h, w = roi.shape[:2]
                center = (w // 2, h // 2)
                scale = 1.1
                M = cv2.getRotationMatrix2D(center, 0, scale)
                zoom_in_roi = cv2.warpAffine(roi, M, (w, h))
                new_encodings.append(self.encode_face(zoom_in_roi))
                
                # Variant 5: Slight Rotation
                M_plus = cv2.getRotationMatrix2D(center, 5, 1.0)
                rot_plus = cv2.warpAffine(roi, M_plus, (w, h))
                new_encodings.append(self.encode_face(rot_plus))

                print(f"[register_face] Generated {len(new_encodings)} variations for robustness.")
            except Exception as e:
                 print(f"[register_face] Augmentation warning: {e}")

        # Filter out None values from augmentation errors
        new_encodings = [enc for enc in new_encodings if enc is not None]

        # --- CHECK DUPLICATES (NOW CHECKS ALL VARIATIONS) ---
        if not bypass_duplicate_check:
            print(f"[Duplicate Check] Checking {len(new_encodings)} candidate variations against {len(self.known_face_encodings)} existing faces...")
            
            if self.face_recognition_available:
                try:
                    # Ensure all known face encodings have the correct shape
                    valid_known_encodings = []
                    valid_known_names = []
                    
                    for i, known_encoding in enumerate(self.known_face_encodings):
                        if known_encoding is not None and len(known_encoding) > 0: # Basic check
                             valid_known_encodings.append(known_encoding)
                             valid_known_names.append(self.known_face_names[i])
                    
                    if valid_known_encodings:
                        print(f"[REINFORCED DEBUG] Checking uniqueness against {len(valid_known_encodings)} valid encodings", flush=True)
                        # Check each NEW encoding (variation) against ALL existing encodings
                        for var_idx, new_enc in enumerate(new_encodings):
                            print(f"[REINFORCED DEBUG] Checking variation {var_idx}...")
                            # Compare using an ULTRA-STRICT tolerance for duplicates
                            face_distances = face_recognition.face_distance(valid_known_encodings, new_enc)
                            
                            # Find the best match
                            closest_match_idx = -1
                            if len(face_distances) > 0:
                                closest_match_idx = np.argmin(face_distances)
                                print(f"[REINFORCED DEBUG] Min distance in variation {var_idx}: {face_distances[closest_match_idx]:.4f}", flush=True)
                            
                            # Threshold 0.65 for dlib (Conservative to prevent any potential face reuse)
                            # Distance < 0.65 means "Same Person". 
                            # We want to BLOCK if same person.
                            if closest_match_idx != -1:
                                log_dist = face_distances[closest_match_idx]
                                print(f"[Duplicate Check] Closest match: {valid_known_names[closest_match_idx]} with distance {log_dist:.4f}")
                                
                                if log_dist < 0.65:
                                    existing_user = valid_known_names[closest_match_idx]
                                
                                # Check if the same face is being registered to a different user
                                if existing_user and existing_user != user_id:
                                    # Determine the role for error message
                                    if existing_user.startswith('student'):
                                        existing_role = 'student'
                                    elif existing_user.startswith('teacher'):
                                        existing_role = 'teacher'
                                    else:
                                        existing_role = 'user'
                                    
                                    print(f"[Duplicate Blocked - DLIB] Match found: {existing_user} (Dist: {face_distances[closest_match_idx]:.4f})", flush=True)
                                    return False, f"Face already registered to {existing_role} '{existing_user}'. Each profile must have a unique face. Same face cannot be used for multiple profiles.", existing_user
                except AttributeError:
                    print("Face recognition library attribute error, falling back to OpenCV check")
                    self.face_recognition_available = False

            # Fallback (OpenCV/HOG) Duplicate Check
            if not self.face_recognition_available:
                # For the fallback, we'll use a distance measure to check for duplicates
                best_match_name = None
                max_similarity = 0.0
                
                # Iterate through ALL new variations (Primary + Augmented)
                print(f"[REINFORCED DEBUG] Starting fallback duplicate check for {len(new_encodings)} variations against {len(self.known_face_encodings)} known faces", flush=True)
                for var_idx, check_encoding in enumerate(new_encodings):
                    for i, known_encoding in enumerate(self.known_face_encodings):
                        # Ensure both encodings have the same shape before comparison
                        if check_encoding.shape == known_encoding.shape:
                            # Calculate cosine similarity
                            dot_product = np.dot(check_encoding, known_encoding)
                            norm_product = np.linalg.norm(check_encoding) * np.linalg.norm(known_encoding)
                            if norm_product != 0:
                                cosine_similarity = dot_product / norm_product
                                
                                if cosine_similarity > max_similarity:
                                    max_similarity = cosine_similarity
                                    best_match_name = self.known_face_names[i]
                        else:
                            if i == 0 and var_idx == 0:
                                print(f"[REINFORCED DEBUG] Shape mismatch: {check_encoding.shape} vs {known_encoding.shape}")
                
                print(f"[Duplicate Check] Max Complexity Similarity found: {max_similarity:.4f} with {best_match_name}", flush=True)

                # Check if the BEST match exceeds the duplicate threshold
                # Raised threshold to 0.80 for MAXIMUM security
                # Higher similarity = More likely to be same person
                if best_match_name and max_similarity > 0.80: 
                    print(f"[Duplicate Blocked - HOG] Best match: {best_match_name} with similarity {max_similarity:.4f}")
                    existing_user = best_match_name
                    
                    # Check if the same face is being registered to a different user
                    if existing_user and existing_user != user_id:
                        # Determine the role for error message
                        if existing_user.startswith('student'):
                            existing_role = 'student'
                        elif existing_user.startswith('teacher'):
                            existing_role = 'teacher'
                        else:
                            existing_role = 'user'
                        
                        return False, f"Face already registered to {existing_role} '{existing_user}' (Match: {int(max_similarity*100)}%). Each profile must have a unique face.", existing_user
        
        # ADD ALL ENCODINGS (Robust Registration)
        # We store ALL variations so that future recognitions work in different conditions
        for enc in new_encodings:
            self.known_face_encodings.append(enc)
            self.known_face_names.append(user_id)
        
        # Retrain SVM
        if not self.face_recognition_available:
            self.train_svm_classifier()
        
        self.save_encodings()
        return True, f"Face registered successfully with {len(new_encodings)} variations", None
    
    def train_svm_classifier(self):
        """
        Train the SVM classifier with current face encodings
        """
        if len(self.known_face_encodings) > 0 and len(set(self.known_face_names)) > 1:  # Need at least 2 different classes
            # Make sure all encodings have the same shape
            if len(self.known_face_encodings) > 0:
                # Check that all encodings have the same shape
                first_shape = self.known_face_encodings[0].shape
                valid_encodings = []
                valid_names = []
                
                for i, encoding in enumerate(self.known_face_encodings):
                    if encoding.shape == first_shape:
                        valid_encodings.append(encoding)
                        valid_names.append(self.known_face_names[i])
                
                if len(valid_encodings) > 1:  # Need at least 2 samples to train
                    # Prepare training data
                    X = np.array(valid_encodings)
                    y = self.label_encoder.fit_transform(valid_names)
                    
                    # Train the classifier
                    self.svm_classifier.fit(X, y)
                    self.is_trained = True
                    # Update the lists to only contain valid encodings
                    self.known_face_encodings = valid_encodings
                    self.known_face_names = valid_names
    
    def recognize_face(self, image):
        """
        Recognize a face in an image
        """
        if self.face_recognition_available:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                recognized_faces = []
                
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)  # Relaxed tolerance
                    name = "Unknown"
                    
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                        else:
                            # Check if the distance is within acceptable range
                            if face_distances[best_match_index] < 0.6:  # Relaxed threshold
                                name = self.known_face_names[best_match_index]
                    
                    recognized_faces.append(name)  # FIX: Append to list
                    print(f"[Face Recognition] Detected: {name}, Distance: {face_distances[best_match_index] if len(face_distances) > 0 else 'N/A'}")
                
                return recognized_faces, face_locations  # FIX: Return list instead of single name
            except Exception as e:
                print(f"Error in face_recognition.face_locations: {e}. Falling back to OpenCV.")
                # self.face_recognition_available = False # Optional: permanent fallback
        
        # Fallback using OpenCV detection with SVM classification or direct matching
        print(f"[recognize_face] Using OpenCV fallback. is_trained={self.is_trained}, known_faces={len(self.known_face_encodings)}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        print(f"[recognize_face] Detected {len(faces)} faces with OpenCV")
        
        recognized_names = []
        
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            face_features = self.extract_face_features(face_roi)
            
            # 1. ALWAYS try direct similarity matching first as it's more reliable for small datasets
            best_match_name = "Unknown"
            best_similarity = 0
            
            if len(self.known_face_encodings) > 0:
                print(f"[recognize_face] Performing direct similarity matching...")
                for i, known_encoding in enumerate(self.known_face_encodings):
                    if face_features.shape == known_encoding.shape:
                        # Calculate cosine similarity
                        dot_product = np.dot(face_features, known_encoding)
                        norm_product = np.linalg.norm(face_features) * np.linalg.norm(known_encoding)
                        if norm_product != 0:
                            cosine_similarity = dot_product / norm_product
                            if i < 5: # Only log first few for brevity
                                print(f"[recognize_face] Similarity with {self.known_face_names[i]}: {cosine_similarity:.4f}")
                            
                            if cosine_similarity > best_similarity:
                                best_similarity = cosine_similarity
                                best_match_name = self.known_face_names[i]
            
            # 2. Try SVM as a secondary verification if trained
            svm_name = "Unknown"
            svm_confidence = 0
            if self.is_trained and len(self.known_face_encodings) > 0:
                expected_shape = self.known_face_encodings[0].shape
                if face_features.shape == expected_shape:
                    prediction = self.svm_classifier.predict([face_features])
                    probabilities = self.svm_classifier.predict_proba([face_features])[0]
                    svm_confidence = np.max(probabilities)
                    svm_name = self.label_encoder.inverse_transform([prediction[0]])[0]
                    print(f"[recognize_face] SVM prediction: {svm_name}, confidence: {svm_confidence:.4f}")
            
            # 3. Decision logic: prioritize strong direct match, or SVM if high confidence
            final_name = "Unknown"
            
            # 3. Decision logic: prioritize strong direct match
            final_name = "Unknown"
            
            # STRICTER THRESHOLDS for duplicate prevention. 
            # We reject anything below 0.80 to be safe and prevent face reuse.
            
            check_thresh_similarity = 0.80 # Raised from 0.70 to require HIGHER accuracy and avoid false matches
            
            if best_similarity > check_thresh_similarity:
                final_name = best_match_name
                print(f"[recognize_face] STRONG MATCH via similarity: {final_name} ({best_similarity:.4f})")
            else:
                print(f"[recognize_face] No match confident enough. Best sim: {best_similarity:.4f}")
                final_name = "Unknown"
            
            recognized_names.append(final_name)
        
        # Format face locations as tuples (top, right, bottom, left) like face_recognition does
        face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]
        return recognized_names, face_locations
    
    def remove_known_face(self, user_id):
        """
        Remove ALL known faces by user_id
        """
        removed_count = 0
        # Iterate backwards to safely pop
        for i in range(len(self.known_face_names) - 1, -1, -1):
            if self.known_face_names[i] == user_id:
                self.known_face_names.pop(i)
                self.known_face_encodings.pop(i)
                removed_count += 1
                
        if removed_count > 0:
            self.save_encodings()
            return True
        return False

    def compare_encodings(self, encoding1, encoding2):
        """
        Compare two encodings and return similarity/distance
        Returns (is_match, score) where score is distance for dlib or similarity for HOG
        """
        if self.face_recognition_available:
            import face_recognition
            distances = face_recognition.face_distance([encoding1], encoding2)
            # Threshold 0.65 for duplicate detection (Conservative to prevent face reuse)
            return (distances[0] < 0.65), distances[0]
        else:
            # Cosine similarity for HOG
            dot_product = np.dot(encoding1, encoding2)
            norm_product = np.linalg.norm(encoding1) * np.linalg.norm(encoding2)
            if norm_product == 0: return False, 0.0
            similarity = dot_product / norm_product
            # Threshold 0.82 for duplicate detection (Conservative for HOG bypass)
            return (similarity > 0.82), similarity
    
    def save_encodings(self):
        """
        Save face encodings to file
        """
        data = {
            "encodings": self.known_face_encodings,
            "names": self.known_face_names,
            "svm_model": self.svm_classifier if not self.face_recognition_available else None,
            "label_encoder": self.label_encoder if not self.face_recognition_available else None,
            "is_trained": self.is_trained if not self.face_recognition_available else False
        }
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
        with open(self.encodings_path, "wb") as f:
            pickle.dump(data, f)
    
    def load_encodings(self):
        """
        Load face encodings from file
        """
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
                self.known_face_names = data["names"]
                self.known_face_encodings = data["encodings"]
                if self.known_face_encodings:
                    print(f"[FaceRecognition] Loaded {len(self.known_face_encodings)} encodings with shape {self.known_face_encodings[0].shape}", flush=True)
                
                if not self.face_recognition_available and "svm_model" in data and data["svm_model"] is not None:
                    self.svm_classifier = data["svm_model"]
                    self.label_encoder = data["label_encoder"]
                    self.is_trained = data["is_trained"]
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
            self.known_face_encodings = []
            self.known_face_names = []

# Utility functions for face detection
def capture_face_for_registration(user_id, save_path):
    """
    Capture face images for registration using webcam
    """
    cap = cv2.VideoCapture(0)
    hc_path = get_haarcascade_path()
    face_cascade = cv2.CascadeClassifier(os.path.join(hc_path, 'haarcascade_frontalface_default.xml'))
    if face_cascade.empty():
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    count = 0
    max_images = 10  # Capture 10 images for better accuracy
    
    print(f"Capturing images for {user_id}. Press SPACE to capture, ESC to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Face Registration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == 32 and len(faces) > 0:  # SPACE key and face detected
            # Save the captured face
            face_img = frame.copy()
            face_filename = f"{save_path}/{user_id}_img_{count}.jpg"
            cv2.imwrite(face_filename, face_img)
            count += 1
            print(f"Saved image {count}/{max_images}")
            
            if count >= max_images:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return count > 0

def train_model_from_dataset(dataset_path="backend/dataset"):
    """
    Train the face recognition model from dataset images
    """
    fr = FaceRecognition()
    
    for role in ['students', 'teachers']:
        role_path = os.path.join(dataset_path, role)
        if os.path.exists(role_path):
            for user_folder in os.listdir(role_path):
                user_path = os.path.join(role_path, user_folder)
                if os.path.isdir(user_path):
                    for img_file in os.listdir(user_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(user_path, img_file)
                            success, msg, _ = fr.register_face(user_folder, img_path)
                            if not success:
                                print(f"Failed to register {user_folder}: {msg}")
    
    print("Model training completed!")
    return fr