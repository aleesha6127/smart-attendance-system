import cv2
import numpy as np
import os

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

def capture_multiple_frames_for_registration(user_id, save_path, num_frames=5):
    """
    Capture multiple face images for registration using webcam for better accuracy
    Returns: (success, message, captured_count)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Could not access webcam", 0
    
    hc_path = get_haarcascade_path()
    face_cascade = cv2.CascadeClassifier(os.path.join(hc_path, 'haarcascade_frontalface_default.xml'))
    if face_cascade.empty():
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    captured_count = 0
    frame_quality_scores = []
    
    print(f"Capturing {num_frames} images for {user_id}. Press SPACE to capture each frame, ESC to exit.")
    
    os.makedirs(save_path, exist_ok=True)
    
    while captured_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame for natural interaction
        frame_display = cv2.flip(frame, 1)
        
        # Detect faces in current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        # Draw face rectangles and quality indicators
        for (x, y, w, h) in faces:
            # Calculate face quality for this detection
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                # Simple quality check
                brightness = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
                contrast = np.std(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
                sharpness = cv2.Laplacian(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                
                # Quality score (0-100)
                quality_score = min(100, 
                                  (brightness/255 * 30) + 
                                  (min(contrast, 50)/50 * 30) + 
                                  (min(sharpness, 200)/200 * 40))
                
                # Draw rectangle with color based on quality
                if quality_score > 80:
                    color = (0, 255, 0)  # Green - Good quality
                elif quality_score > 60:
                    color = (0, 255, 255)  # Yellow - Acceptable
                else:
                    color = (0, 0, 255)  # Red - Poor quality
                
                cv2.rectangle(frame_display, (frame.shape[1]-x-w, y), (frame.shape[1]-x, y+h), color, 2)
                cv2.putText(frame_display, f"Q: {quality_score:.0f}%", 
                           (frame.shape[1]-x-w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        # Add instructions
        cv2.putText(frame_display, f"Frame {captured_count+1}/{num_frames} - Press SPACE to capture", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, "ESC to cancel", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        cv2.imshow('Multi-Frame Face Registration', frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == 32 and len(faces) > 0:  # SPACE key and face detected
            # Select the best face (largest)
            best_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = best_face
            
            # Save the captured face
            face_img = frame.copy()
            face_filename = f"{save_path}/{user_id}_frame_{captured_count}.jpg"
            cv2.imwrite(face_filename, face_img)
            
            # Calculate and store quality score
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                brightness = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
                contrast = np.std(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
                sharpness = cv2.Laplacian(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                quality_score = min(100, 
                                  (brightness/255 * 30) + 
                                  (min(contrast, 50)/50 * 30) + 
                                  (min(sharpness, 200)/200 * 40))
                frame_quality_scores.append(quality_score)
                print(f"Saved frame {captured_count+1}/{num_frames} (Quality: {quality_score:.1f}%)")
            
            captured_count += 1
            
            # Brief pause to allow user to adjust
            if captured_count < num_frames:
                cv2.waitKey(1000)  # 1 second delay
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured_count > 0:
        avg_quality = sum(frame_quality_scores) / len(frame_quality_scores) if frame_quality_scores else 0
        return True, f"Captured {captured_count} frames with average quality {avg_quality:.1f}%", captured_count
    else:
        return False, "No frames captured", 0

def capture_face_for_registration(user_id, save_path):
    """
    Capture face images for registration using webcam (single frame version)
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