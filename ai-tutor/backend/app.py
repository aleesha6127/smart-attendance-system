from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file, send_from_directory
import sqlite3
import hashlib
import os
import pickle
import cv2
import numpy as np
import base64
from datetime import datetime, timedelta
import json
from werkzeug.exceptions import RequestEntityTooLarge
from capture_utils import capture_multiple_frames_for_registration, capture_face_for_registration
from ai_tutor import AITutor
from avatar_utils import generate_avatar, get_default_avatar_url
from xhtml2pdf import pisa
import io
from face_recognition_module import FaceRecognition
from sms_utils import send_credentials_sms

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# Twilio Configuration (Set these as environment variables)
os.environ['TWILIO_ACCOUNT_SID'] = os.getenv('TWILIO_ACCOUNT_SID', 'your_account_sid_here')
os.environ['TWILIO_AUTH_TOKEN'] = os.getenv('TWILIO_AUTH_TOKEN', 'your_auth_token_here')
os.environ['TWILIO_PHONE_NUMBER'] = os.getenv('TWILIO_PHONE_NUMBER', 'your_phone_number_here')

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Initialize face recognition and AI tutor
# Use correct path relative to backend directory (where app.py runs from)
face_recognizer = FaceRecognition(encodings_path='models/face_encodings.pkl')
ai_tutor = AITutor()

# LBPH Recognizer - More accurate for face matching
try:
    from lbph_recognizer import lbph_recognizer
    USE_LBPH = lbph_recognizer.is_trained
    print(f"[+] LBPH Recognizer loaded. is_trained={USE_LBPH}")
except Exception as e:
    print(f"LBPH not available: {e}")
    USE_LBPH = False

def get_db_connection():
    conn = sqlite3.connect('database/attendance.db')
    conn.row_factory = sqlite3.Row
    return conn

# ============================================
# Attendance Governance Helper Functions
# ============================================

# Face recognition confidence threshold
CONFIDENCE_THRESHOLD = 75.0  # Auto-confirm if >= 75%

def log_attendance_change(conn, attendance_id, action, old_status, new_status, 
                          changed_by, editor_role, justification=None, confidence_score=None):
    """Log all attendance changes for audit trail"""
    conn.execute('''
        INSERT INTO attendance_audit_log 
        (attendance_id, action, old_status, new_status, changed_by, editor_role, justification, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (attendance_id, action, old_status, new_status, changed_by, editor_role, justification, confidence_score))

def check_attendance_editable(attendance_record, user_role):
    """
    Check if an attendance record can be edited.
    Returns (can_edit, reason)
    """
    if not attendance_record:
        return False, "Record not found"
    
    # Admin can always edit (override)
    if user_role == 'admin':
        return True, "Admin override"
    
    # Check if record is locked
    if attendance_record['is_locked']:
        return False, "Record is locked. Contact admin for changes."
    
    # Check if it's the same day (teachers can only edit same-day records)
    if user_role == 'teacher':
        record_date = attendance_record['date']
        today = datetime.now().strftime('%Y-%m-%d')
        if record_date != today:
            return False, "Can only edit attendance on the same day it was recorded."
    
    return True, "Editable"

def auto_lock_old_attendance(conn):
    """
    Lock attendance records older than 24 hours.
    Uses the date column since records are locked after their date has passed.
    """
    try:
        conn.execute('''
            UPDATE attendance 
            SET is_locked = 1, locked_at = CURRENT_TIMESTAMP
            WHERE is_locked = 0 
            AND date < date('now', '-1 day')
        ''')
        conn.commit()
    except Exception as e:
        # If columns don't exist yet, silently continue
        print(f"Auto-lock skipped: {e}")

def get_attendance_status_options():
    """Return valid attendance status options"""
    return ['present', 'late', 'excused', 'absent']

@app.route('/')
def index():
    if 'user_id' in session:
        if session['role'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif session['role'] == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        elif session['role'] == 'student':
            return redirect(url_for('student_dashboard'))
    return render_template('landing_page.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        role = request.form.get('role')  # Get the role from the form
        
        # Hash the password for comparison
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE user_id = ? AND password_hash = ? AND is_active = 1",
            (user_id, password_hash)
        ).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['user_id']
            session['role'] = user['role']
            session['name'] = user['name']
            session['avatar'] = user['avatar']  # Include avatar in session
            session['face_image_path'] = user['face_image_path']  # Include face image path
            
            return redirect(url_for(user['role'] + '_dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials or user does not exist')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ============================================
# Public Registration & Admin Approval Routes
# ============================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Public registration form with personal details and face capture"""
    if request.method == 'POST':
        # Handle form submission and biometric capture
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        role = request.form.get('role')
        department = request.form.get('department')
        batch = request.form.get('batch')
        parent_name = request.form.get('parent_name')
        parent_phone = request.form.get('parent_phone')
        image_data = request.form.get('image_data')

        if not all([name, email, role, image_data]):
            flash("Missing required fields.", "error")
            return render_template('register.html', form_data=request.form)

        # --- SERVER SIDE VALIDATION ---
        import re
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        phone_regex = r'^[0-9]{10}$'

        if not re.match(email_regex, email):
            flash("Invalid email format.", "error")
            return render_template('register.html', form_data=request.form)
        
        if not re.match(phone_regex, phone):
            flash("Invalid phone number. Must be 10 digits.", "error")
            return render_template('register.html', form_data=request.form)
        
        if role == 'student' and parent_phone and not re.match(phone_regex, parent_phone):
            flash("Invalid parent phone number. Must be 10 digits.", "error")
            return render_template('register.html', form_data=request.form)

        # Check for existing email or phone in users or pending_registrations
        conn = get_db_connection()
        existing_user_email = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        existing_pending_email = conn.execute("SELECT 1 FROM pending_registrations WHERE email = ? AND registration_status = 'pending'", (email,)).fetchone()
        
        existing_user_phone = conn.execute("SELECT 1 FROM users WHERE phone = ?", (phone,)).fetchone()
        existing_pending_phone = conn.execute("SELECT 1 FROM pending_registrations WHERE phone = ? AND registration_status = 'pending'", (phone,)).fetchone()
        
        if existing_user_email or existing_pending_email:
            conn.close()
            flash("Email already registered or pending approval.", "error")
            return render_template('register.html', form_data=request.form)
            
        if existing_user_phone or existing_pending_phone:
            conn.close()
            flash("Phone number already registered or pending approval.", "error")
            return render_template('register.html', form_data=request.form)

        # Process and save face image
        try:
            image_data_parts = image_data.split(',')
            if len(image_data_parts) < 2:
                flash("Invalid image data.", "error")
                return render_template('register.html', form_data=request.form)
            
            image_data_encoded = image_data_parts[1]
            image_bytes = base64.b64decode(image_data_encoded)
            
            # Save to temporary folder for pending approvals
            os.makedirs('dataset/pending/', exist_ok=True)
            filename = f"pending_{email.split('@')[0]}_{int(datetime.now().timestamp())}.jpg"
            image_path = os.path.join('dataset/pending/', filename)
            
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            # --- BIOMETRIC DUPLICATE CHECK ---
            # 1. Check against active system (Students & Teachers)
            image_bgr = cv2.imread(image_path)
            if image_bgr is not None:
                recognized_names, _ = face_recognizer.recognize_face(image_bgr)
                known_faces = [name for name in recognized_names if name != "Unknown"]
                
                if known_faces:
                    duplicate_user_id = known_faces[0]
                    if os.path.exists(image_path): os.remove(image_path)
                    conn.close()
                    flash(f"Failed to submit the registration form. Face is already registered to user ID {duplicate_user_id}.", "error")
                    return render_template('register.html', form_data=request.form)
                
                # 2. Check against OTHER pending registrations (no encoding yet)
                # This ensures two people don't register with the same face before approval
                new_face_encoding = face_recognizer.encode_face(image_bgr)
                
                if new_face_encoding is not None:
                    pending_requests = conn.execute("SELECT face_image_path FROM pending_registrations WHERE registration_status = 'pending'").fetchall()
                    for req in pending_requests:
                        other_path = req['face_image_path']
                        if not os.path.exists(other_path) or other_path == image_path:
                            continue # Skip self or missing files
                        
                        other_img_bgr = cv2.imread(other_path)
                        if other_img_bgr is not None:
                            other_encoding = face_recognizer.encode_face(other_img_bgr)
                            if other_encoding is not None:
                                is_duplicate, score = face_recognizer.compare_encodings(other_encoding, new_face_encoding)
                                if is_duplicate:
                                    if os.path.exists(image_path): os.remove(image_path)
                                    conn.close()
                                    flash(f"Failed to submit the registration form. This face matches another pending registration request (Score: {score:.4f}).", "error")
                                    return render_template('register.html', form_data=request.form)
            
            # Store in pending_registrations
            import sqlite3
            try:
                conn.execute("""
                    INSERT INTO pending_registrations 
                    (name, email, phone, department, batch, role, parent_name, parent_phone, face_image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, email, phone, department, batch, role, parent_name, parent_phone, image_path))
                conn.commit()
            except sqlite3.IntegrityError:
                if os.path.exists(image_path): os.remove(image_path)
                conn.close()
                flash(f"Failed to submit the registration form. The email '{email}' is already registered or pending approval.", "error")
                return render_template('register.html', form_data=request.form)
            
            conn.close()
            
            flash("Registration submitted successfully! Please wait for admin approval.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            if 'conn' in locals(): conn.close()
            flash(f"An error occurred: {str(e)}", "error")
            return render_template('register.html', form_data=request.form)

    return render_template('register.html')

@app.route('/admin/approvals')
def admin_approvals():
    """Admin dashboard for reviewing pending registrations"""
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    approvals = conn.execute("SELECT * FROM pending_registrations WHERE registration_status = 'pending' ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template('admin/approvals.html', approvals=approvals)

@app.route('/admin/approve_registration/<int:reg_id>', methods=['POST'])
def admin_approve_registration(reg_id):
    """Approve a registration and generate user credentials"""
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    reg = conn.execute("SELECT * FROM pending_registrations WHERE id = ?", (reg_id,)).fetchone()
    
    if not reg:
        conn.close()
        flash("Registration not found.", "error")
        return redirect(url_for('admin_approvals'))
    
    try:
        # Generate User ID based on the highest existing ID for that role
        role = reg['role']
        
        # Get the highest existing ID for this role to avoid UNIQUE constraint failures if users were deleted
        highest_id_record = conn.execute(
            f"SELECT user_id FROM users WHERE role = ? ORDER BY user_id DESC LIMIT 1", 
            (role,)
        ).fetchone()
        
        if highest_id_record:
            # Extract the numeric part (e.g., 'student005' -> 5)
            highest_id = highest_id_record['user_id']
            # Strip the role prefix
            try:
                numeric_part = int(highest_id.replace(role, ''))
                next_num = numeric_part + 1
            except ValueError:
                # Fallback if ID format is unexpected
                count = conn.execute(f"SELECT COUNT(*) as count FROM users WHERE role = ?", (role,)).fetchone()['count']
                next_num = count + 1
        else:
            # First user for this role
            next_num = 1
            
        user_id = f"{role}{str(next_num).zfill(3)}"
        
        # Generate random password
        import string
        import random
        password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Move image to permanent record
        save_dir = f"dataset/{role}s/{user_id}/"
        os.makedirs(save_dir, exist_ok=True)
        final_filename = f"{user_id}_{int(datetime.now().timestamp())}.jpg"
        final_image_path = os.path.join(save_dir, final_filename)
        
        if os.path.exists(reg['face_image_path']):
            os.rename(reg['face_image_path'], final_image_path)
        
        db_face_path = f"{user_id}/{final_filename}"
        
        # Generate avatar
        avatar_path = generate_avatar(reg['name'])
        
        # Create user
        conn.execute("""
            INSERT INTO users (user_id, name, email, password_hash, role, department, batch, avatar, face_image_path, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (user_id, reg['name'], reg['email'], password_hash, role, reg['department'], reg['batch'], avatar_path, db_face_path))
        
        # Update registration status
        conn.execute("UPDATE pending_registrations SET registration_status = 'approved' WHERE id = ?", (reg_id,))
        conn.commit()
        conn.close()
        
        # Immediate Biometric Sync: Index the face with augmentation for robust recognition
        # This ensures the user is immediately searchable for duplicate checks
        face_recognizer.register_face(user_id, final_image_path)
        if USE_LBPH:
            try:
                from lbph_recognizer import train_lbph_model
                train_lbph_model()
            except:
                pass

        # Send SMS with credentials
        sms_sent, sms_msg = send_credentials_sms(reg['phone'], user_id, password, role)
        print(f"[NOTIFICATION] User {user_id} approved. Credentials sent to {reg['phone']}: {sms_msg}")
        
        if sms_sent:
            flash(f"Registration approved! Credentials securely sent to user via SMS.", "success")
        else:
            flash(f"Registration approved! User ID: {user_id}, Password: {password} (SMS Failed: {sms_msg})", "warning")
            
        return redirect(url_for('admin_approvals'))
    except Exception as e:
        if 'conn' in locals(): conn.close()
        flash(f"Error during approval: {str(e)}", "error")
        return redirect(url_for('admin_approvals'))

@app.route('/admin/reject_registration/<int:reg_id>', methods=['POST'])
def admin_reject_registration(reg_id):
    """Reject a registration and delete temporary data"""
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    reg = conn.execute("SELECT * FROM pending_registrations WHERE id = ?", (reg_id,)).fetchone()
    
    if reg:
        if os.path.exists(reg['face_image_path']):
            os.remove(reg['face_image_path'])
        
        conn.execute("DELETE FROM pending_registrations WHERE id = ?", (reg_id,))
        conn.commit()
        flash("Registration rejected and data deleted.", "info")
    
    conn.close()
    return redirect(url_for('admin_approvals'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get counts for dashboard (Registered/Active users only)
    total_students = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'student' AND is_active = 1").fetchone()['count']
    total_teachers = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'teacher' AND is_active = 1").fetchone()['count']
    total_admins = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'admin'").fetchone()['count']
    
    # Get recent activities (Registered/Active users only)
    recent_users = conn.execute("SELECT * FROM users WHERE role != 'admin' AND is_active = 1 ORDER BY id DESC LIMIT 5").fetchall()
    
    # Calculate real overall attendance rate from actual attendance records
    # Leave status counts as 0.5 attendance credit
    attendance_stats = conn.execute("""
        SELECT 
            COUNT(*) as total_records,
            SUM(CASE WHEN status = 'present' THEN 1 
                     WHEN status = 'leave' THEN 0.5 
                     ELSE 0 END) as attended_credit
        FROM attendance
    """).fetchone()
    
    if attendance_stats['total_records'] > 0:
        overall_attendance_rate = round((attendance_stats['attended_credit'] / attendance_stats['total_records']) * 100, 1)
    else:
        overall_attendance_rate = 0
    
    # Count recent activities
    recent_activities_count = len(recent_users)
    
    low_attendance_students = 0
    if total_students > 0:
        # High performers are those WITH attendance records >= 85%
        high_performers_count = conn.execute("""
            SELECT COUNT(*) as count FROM (
                SELECT student_id
                FROM attendance
                GROUP BY student_id
                HAVING (SUM(CASE WHEN status = 'present' THEN 1 
                                 WHEN status = 'leave' THEN 0.5 
                                 ELSE 0 END) * 100.0 / COUNT(*)) >= 85
            ) AS high_perf
        """).fetchone()['count']
        
        # Everyone else (including those with no records) needs attention
        low_attendance_students = total_students - high_performers_count
    
    # Count pending requests
    pending_requests = conn.execute("SELECT COUNT(*) as count FROM attendance_requests WHERE status = 'pending'").fetchone()['count']
    
    conn.close()
    
    # Pass individual counts and calculated total (excluding admin for user preference)
    total_non_admins = total_students + total_teachers
    
    return render_template('admin/dashboard.html', 
                          total_students=total_students, 
                          total_teachers=total_teachers, 
                          total_admins=total_admins,
                          total_members=total_non_admins,
                          recent_users=recent_users,
                          overall_attendance_rate=overall_attendance_rate,
                          recent_activities_count=recent_activities_count,
                          low_attendance_students=low_attendance_students,
                          pending_requests=pending_requests)

def get_user_face_image_path(user_id, role):
    """
    Get the path to the user's face image
    """
    import os
    import glob
    
    # Define the save directory based on user role
    if role == 'student':
        save_dir = f"dataset/students/{user_id}/"
    elif role == 'teacher':
        save_dir = f"dataset/teachers/{user_id}/"
    else:
        save_dir = f"dataset/others/{user_id}/"
    
    # Check if the directory exists
    if os.path.exists(save_dir):
        # Find all jpg files in the directory
        image_files = glob.glob(os.path.join(save_dir, f"{user_id}_*.jpg"))
        if image_files:
            # Return the first image file found (most recent one)
            # Return path relative to the dataset/{role} folder
            full_path = image_files[0]
            # Replace backslashes for URL compatibility
            rel_path = os.path.relpath(full_path, os.path.dirname(os.path.dirname(save_dir))).replace('\\', '/')
            # Strip the 'students/', 'teachers/', or 'others/' part to get just 'user_id/filename'
            if rel_path.startswith(('students/', 'teachers/', 'others/')):
                rel_path = '/'.join(rel_path.split('/')[1:])
            return rel_path
    
    return None

@app.route('/admin/users')
def admin_users():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    # Only show non-admin active users (who have completed face registration)
    users = conn.execute("SELECT * FROM users WHERE role != 'admin' AND is_active = 1 ORDER BY role, name").fetchall()
    conn.close()
    
    # Separate students and teachers
    students = [user for user in users if user['role'] == 'student']
    teachers = [user for user in users if user['role'] == 'teacher']
    
    # Add face image paths to users
    for user in students:
        user_dict = dict(user)
        user_dict['face_image_path'] = get_user_face_image_path(user_dict['user_id'], user_dict['role'])
        students[students.index(user)] = user_dict
    
    for user in teachers:
        user_dict = dict(user)
        user_dict['face_image_path'] = get_user_face_image_path(user_dict['user_id'], user_dict['role'])
        teachers[teachers.index(user)] = user_dict
    
    return render_template('admin/users.html', students=students, teachers=teachers)

@app.route('/admin/add_user', methods=['GET', 'POST'])
def admin_add_user():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        user_id = request.form['user_id']
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        department = request.form.get('department')
        batch = request.form.get('batch')
        semester = request.form.get('semester')
        phone = request.form.get('phone')
        parent_name = request.form.get('parent_name')
        parent_phone = request.form.get('parent_phone')
        
        # DEBUG: Print all form data
        print(f"[ADD_USER DEBUG] user_id={user_id}, name={name}, email={email}, role={role}")
        print(f"[ADD_USER DEBUG] department={department}, batch={batch}, semester={semester}")
        
        # For teachers and admins, make department, batch, and semester optional
        if role in ['teacher', 'admin']:
            department = department if department else None
            batch = batch if batch else None
            semester = semester if semester else None
        
        # Email validation
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            flash('Invalid email format. Please enter a valid email address.', 'error')
            return render_template('admin/add_user.html')
        
        # User ID prefix validation - ensure ID prefix matches role
        user_id_lower = user_id.lower()
        if role == 'student' and not user_id_lower.startswith('student'):
            flash('Error: Student User ID must start with "student" (e.g., student001, student002)', 'error')
            return render_template('admin/add_user.html')
        elif role == 'teacher' and not user_id_lower.startswith('teacher'):
            flash('Error: Teacher User ID must start with "teacher" (e.g., teacher001, teacher002)', 'error')
            return render_template('admin/add_user.html')
        elif role == 'admin' and not user_id_lower.startswith('admin'):
            flash('Error: Admin User ID must start with "admin" (e.g., admin001, admin002)', 'error')
            return render_template('admin/add_user.html')

        # Phone validation
        phone_regex = r'^[0-9]{10}$'
        if phone and not re.match(phone_regex, phone):
            flash('Invalid phone number. Must be 10 digits.', 'error')
            return render_template('admin/add_user.html')
        
        if role == 'student' and parent_phone and not re.match(phone_regex, parent_phone):
            flash('Invalid parent phone number. Must be 10 digits.', 'error')
            return render_template('admin/add_user.html')
        
        # Hash the password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = get_db_connection()
        
        try:
            # Check if User ID already exists
            existing_user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            
            if existing_user:
                if existing_user['is_active'] == 0:
                    # If user exists but is inactive (failed registration), update info and resume
                    conn.execute(
                        """UPDATE users SET name=?, email=?, password_hash=?, role=?, department=?, batch=?, semester=?, phone=?, parent_name=?, parent_phone=?
                           WHERE user_id=?""",
                        (name, email, password_hash, role, department, batch, semester, phone, parent_name, parent_phone, user_id)
                    )
                    conn.commit()
                    conn.close()
                    flash('Resuming incomplete registration for this User ID.', 'info')
                    return redirect(url_for('admin_register_face', user_id=user_id))
                else:
                    conn.close()
                    return render_template('admin/add_user.html', error='User ID already exists')

            # Check if email already exists
            existing_email = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
            
            if existing_email:
                conn.close()  # Close connection in case of error
                flash(f'Email address {email} is already registered to another user ({existing_email["name"]}). Email addresses must be unique across all users.', 'error')
                return render_template('admin/add_user.html')
            
            # Check if password is already in use (for strict uniqueness)
            existing_password = conn.execute(
                "SELECT * FROM users WHERE password_hash = ?", (password_hash,)
            ).fetchone()
            
            if existing_password:
                conn.close()
                flash(f'SECURITY ALERT: This password is already in use by another profile. For security and uniqueness, please choose a different password.', 'error')
                return render_template('admin/add_user.html')

            # Generate avatar based on user's name
            avatar_path = generate_avatar(name)
            
            conn.execute(
                """INSERT INTO users (user_id, name, email, password_hash, role, department, batch, semester, phone, parent_name, parent_phone, avatar, is_active) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, name, email, password_hash, role, department, batch, semester, phone, parent_name, parent_phone, avatar_path, 0)
            )
            
            # Commit the transaction so far
            conn.commit()
            
            # Close connection before redirecting to face registration
            conn.close()
            
            # Redirect to face registration page after successful user creation
            return redirect(url_for('admin_register_face', user_id=user_id))
        except sqlite3.Error as e:
            if 'conn' in locals(): conn.close()
            return render_template('admin/add_user.html', error=f'Database error: {e}')
    
    return render_template('admin/add_user.html')

@app.route('/admin/edit_user/<user_id>', methods=['GET', 'POST'])
def admin_edit_user(user_id):
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        role = request.form['role']
        department = request.form.get('department')
        batch = request.form.get('batch')
        semester = request.form.get('semester')
        phone = request.form.get('phone')
        parent_name = request.form.get('parent_name')
        parent_phone = request.form.get('parent_phone')
        is_active = request.form.get('is_active') == 'on'
        
        # Email validation
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            conn.close()
            flash('Invalid email format. Please enter a valid email address.', 'error')
            return render_template('admin/edit_user.html', user=dict(user))
        
        # Phone number validation
        phone_pattern = r'^[0-9]{10}$'
        if phone and phone.strip():
            if not re.match(phone_pattern, phone.strip()):
                conn.close()
                flash('Phone number must be exactly 10 digits.', 'error')
                return render_template('admin/edit_user.html', user=dict(user))
        
        if role == 'student' and parent_phone and parent_phone.strip():
            if not re.match(phone_pattern, parent_phone.strip()):
                conn.close()
                flash('Parent phone number must be exactly 10 digits.', 'error')
                return render_template('admin/edit_user.html', user=dict(user))
        
        # User ID prefix validation - ensure ID prefix matches role
        # First get the current user to check their user_id
        current_user = conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if current_user:
            current_user_id = current_user['user_id']
            user_id_lower = current_user_id.lower()
            
            if role == 'student' and not user_id_lower.startswith('student'):
                conn.close()
                flash('Error: Cannot change role to Student. User ID must start with "student" prefix.', 'error')
                return render_template('admin/edit_user.html', user=dict(user))
            elif role == 'teacher' and not user_id_lower.startswith('teacher'):
                conn.close()
                flash('Error: Cannot change role to Teacher. User ID must start with "teacher" prefix.', 'error')
                return render_template('admin/edit_user.html', user=dict(user))
            elif role == 'admin' and not user_id_lower.startswith('admin'):
                conn.close()
                flash('Error: Cannot change role to Admin. User ID must start with "admin" prefix.', 'error')
                return render_template('admin/edit_user.html', user=dict(user))
        
        # Check if email is already used by another user (excluding current user)
        existing_email = conn.execute(
            "SELECT * FROM users WHERE email = ? AND user_id != ?", (email, user_id)
        ).fetchone()
        
        if existing_email:
            conn.close()
            flash(f'Email address {email} is already registered to another user ({existing_email["name"]}). Email addresses must be unique across all users.', 'error')
            return render_template('admin/edit_user.html', user=dict(user))
        
        conn.execute(
            "UPDATE users SET name=?, email=?, role=?, department=?, batch=?, semester=?, phone=?, parent_name=?, parent_phone=?, is_active=? WHERE user_id=?",
            (name, email, role, department, batch, semester, phone, parent_name, parent_phone, is_active, user_id)
        )
        conn.commit()
        conn.close()
        
        return redirect(url_for('admin_users'))
    
    user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    
    if not user:
        return redirect(url_for('admin_users'))
    
    return render_template('admin/edit_user.html', user=dict(user))

@app.route('/admin/delete_user/<user_id>')
def admin_delete_user(user_id):
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    # Get user role before deleting to know which folder to remove
    conn = get_db_connection()
    user = conn.execute("SELECT role FROM users WHERE user_id = ?", (user_id,)).fetchone()
    
    if user:
        user_role = user['role']
        
        # Delete user from database
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        
        conn.commit()
        
        # Remove face data from face recognition system
        # This will remove the user from the known faces list
        face_recognizer.remove_known_face(user_id)
        
        conn.close()
        
        # Remove the user's face image folder
        import shutil
        import os
        import stat
        if user_role == 'student':
            folder_path = f"dataset/students/{user_id}/"
        elif user_role == 'teacher':
            folder_path = f"dataset/teachers/{user_id}/"
        else:
            folder_path = f"dataset/others/{user_id}/"
        
        if os.path.exists(folder_path):
            # Handle Windows permission errors by setting file permissions
            def handle_remove_readonly(func, path, exc):
                """Handle removing read-only files on Windows"""
                os.chmod(path, stat.S_IWRITE)
                func(path)
            
            try:
                shutil.rmtree(folder_path, onerror=handle_remove_readonly)
            except PermissionError as e:
                print(f"Permission error deleting folder {folder_path}: {e}")
                # Continue execution even if folder deletion fails
    else:
        conn.close()
    
    return redirect(url_for('admin_users'))


@app.route('/admin/register_face/<user_id>', methods=['GET', 'POST'])
def admin_register_face(user_id):
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    # Check if user exists
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        flash('User does not exist', 'error')
        return redirect(url_for('admin_users'))
    
    if request.method == 'POST':
        # Handle face registration
        if 'image_data' in request.form:
            # Validate image data size to prevent large payloads
            image_data = request.form['image_data']
            # Check if the image data is too large (more than ~25MB worth of base64)
            if len(image_data) > 38000000:  # Approximate limit for 25MB base64 encoded image
                flash('Image data is too large. Please capture a smaller image.', 'error')
                return render_template('admin/register_face.html', user_id=user_id, user=dict(user))
            
            # Save the captured image temporarily
            import base64
            import os
            from datetime import datetime
            
            # Decode the image data
            try:
                image_data_parts = image_data.split(',')
                if len(image_data_parts) < 2:
                    flash('Invalid image data format.', 'error')
                    return render_template('admin/register_face.html', user_id=user_id, user=dict(user))
                
                header, image_data_encoded = image_data_parts[0], ','.join(image_data_parts[1:])
                
                # Validate that it's a JPEG image
                if not header.startswith('data:image/jpeg') and not header.startswith('data:image/jpg'):
                    flash('Please use a JPEG image for face registration.', 'error')
                    return render_template('admin/register_face.html', user_id=user_id, user=dict(user))
                
                image_bytes = base64.b64decode(image_data_encoded)
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return render_template('admin/register_face.html', user_id=user_id, user=dict(user))
            
            # Create a temporary image file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_image_path = f"temp_face_{user_id}_{timestamp}.jpg"
            
            with open(temp_image_path, 'wb') as f:
                f.write(image_bytes)
            
            # Get user role to determine where to save the image
            conn = get_db_connection()
            user_data = conn.execute("SELECT role FROM users WHERE user_id = ?", (user_id,)).fetchone()
            user_role = user_data['role'] if user_data else 'other'
            conn.close()
            
            # Define the save directory but don't create it yet
            if user_role == 'student':
                save_dir = f"dataset/students/{user_id}/"
            elif user_role == 'teacher':
                save_dir = f"dataset/teachers/{user_id}/"
            else:
                save_dir = f"dataset/others/{user_id}/"
            
            # Move the temporary image to the appropriate dataset folder
            final_image_path = os.path.join(save_dir, f"{user_id}_{int(datetime.now().timestamp())}.jpg")
            
            # Check for duplicates in the pending folder before registering
            pending_folder = 'dataset/pending/'
            if os.path.exists(pending_folder):
                image_bgr = cv2.imread(temp_image_path)
                new_face_encoding = face_recognizer.encode_face(image_bgr)
                if new_face_encoding is not None:
                    for other_img in os.listdir(pending_folder):
                        other_path = os.path.join(pending_folder, other_img)
                        other_img_bgr = cv2.imread(other_path)
                        if other_img_bgr is not None:
                            other_encoding = face_recognizer.encode_face(other_img_bgr)
                            if other_encoding is not None:
                                is_duplicate, score = face_recognizer.compare_encodings(other_encoding, new_face_encoding)
                                if is_duplicate:
                                    if os.path.exists(temp_image_path): os.remove(temp_image_path)
                                    flash(f"Face registration failed: This face matches a pending registration request (Score: {score:.4f}). Faces must be globally unique.", "error")
                                    return render_template('admin/register_face.html', user_id=user_id, user=dict(user))

            # Register face using the enhanced face recognition module
            success, message, extra_info = face_recognizer.register_face(user_id, temp_image_path)
            
            # Log detailed registration attempt
            print(f"[Registration] Attempt for {user_id}: Success={success}, Message={message}")
            
            if success:
                # Only create directory and move file if registration is successful
                os.makedirs(save_dir, exist_ok=True)
                os.rename(temp_image_path, final_image_path)
                
                # Activate the user and save face image path
                try:
                    # Get the relative path for the DB (teacher002/filename.jpg)
                    db_face_path = os.path.relpath(final_image_path, os.path.dirname(os.path.dirname(save_dir))).replace('\\', '/')
                    if db_face_path.startswith(('students/', 'teachers/', 'others/')):
                        db_face_path = '/'.join(db_face_path.split('/')[1:])
                    
                    conn = get_db_connection()
                    conn.execute("UPDATE users SET is_active = 1, face_image_path = ? WHERE user_id = ?", (db_face_path, user_id))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"Error activating user {user_id}: {e}")
                
                # Retrain LBPH model to include the new face immediately
                if USE_LBPH:
                    try:
                        from lbph_recognizer import train_lbph_model
                        train_lbph_model()
                        print(f"[+] LBPH Model retrained after registering {user_id}")
                    except Exception as e:
                        print(f"Error retraining LBPH: {e}")
                
                flash('Face registered successfully!', 'success')
                return redirect(url_for('admin_users'))
            else:
                # Clean up temp file if registration failed
                if os.path.exists(temp_image_path) and os.path.isfile(temp_image_path):
                    os.remove(temp_image_path)
                
                # Check if the failure is due to duplicate face registration
                if extra_info:
                    # Fetch existing user details to show on the page
                    conn = get_db_connection()
                    matching_user = conn.execute("SELECT * FROM users WHERE user_id = ?", (extra_info,)).fetchone()
                    
                    if matching_user:
                        matching_user = dict(matching_user)
                        matching_user['face_image_path'] = get_user_face_image_path(matching_user['user_id'], matching_user['role'])
                        conn.close()
                        
                        # ATOMIC PURGE: Delete the inactive user since the face is a duplicate
                        # This fulfills the "don't create it" requirement.
                        try:
                            # Handle database connection
                            # We need to re-open because it was closed on line 646 or we can just use the one we already have if it was open
                            conn_purge = get_db_connection()
                            conn_purge.execute("DELETE FROM users WHERE user_id = ? AND is_active = 0", (user_id,))
                            conn_purge.commit()
                            conn_purge.close()
                            print(f"[!] Atomic Purge: Deleted inactive user {user_id} due to duplicate face.")
                        except Exception as e:
                            print(f"Error purging user {user_id}: {e}")

                        flash(f'REGISTRATION ABORTED: This face is already registered to "{matching_user["name"]}". To maintain uniqueness, the profile creation for {user_id} has been canceled.', 'error')
                        return redirect(url_for('admin_users'))
                    conn.close()
                
                # For other registration failures (no face detected, etc.), allow retry on the same page
                flash(f'Registration Error: {message}', 'error')
                return render_template('admin/register_face.html', user_id=user_id, user=dict(user))
        
        flash('No image data received. Please capture an image first.', 'error')
        return render_template('admin/register_face.html', user_id=user_id, user=dict(user))
    
    return render_template('admin/register_face.html', user_id=user_id, user=dict(user))

@app.route('/teacher/dashboard')
def teacher_dashboard():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get teacher-specific data
    teacher_id = session['user_id']
    
    # Get total students assigned to this teacher
    total_students_result = conn.execute(
        "SELECT COUNT(*) as count FROM users WHERE role = 'student'"
    ).fetchone()
    total_students = total_students_result['count'] if total_students_result else 0
    
    # Get overall attendance rate
    # Leave status counts as 0.5 attendance credit
    attendance_result = conn.execute("""
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN status = 'present' THEN 1 
                        WHEN status = 'leave' THEN 0.5 
                        ELSE 0 END) as attended_credit 
        FROM attendance
    """).fetchone()
    total_attendance = attendance_result['total'] if attendance_result['total'] > 0 else 1
    present_count = attendance_result['attended_credit'] or 0
    overall_attendance_rate = round((present_count / total_attendance) * 100, 2)
    
    # Get pending requests
    pending_requests_result = conn.execute(
        "SELECT COUNT(*) as count FROM attendance_requests WHERE status = 'pending'"
    ).fetchone()
    pending_requests = pending_requests_result['count'] if pending_requests_result else 0
    
    # Get total classes taught by this teacher
    total_classes_taught_result = conn.execute(
        "SELECT COUNT(*) as count FROM attendance WHERE marked_by = ?",
        (teacher_id,)
    ).fetchone()
    total_classes_taught = total_classes_taught_result['count'] if total_classes_taught_result else 0
    
    # Get at-risk students (those with low attendance)
    # Leave counts as 0.5 attendance
    at_risk_students_result = conn.execute(
        """
        SELECT u.name, u.user_id, 
               (SUM(CASE WHEN a.status = 'present' THEN 1 
                         WHEN a.status = 'leave' THEN 0.5 
                         ELSE 0 END) * 100.0 / COUNT(a.id)) as attendance_rate
        FROM users u
        LEFT JOIN attendance a ON u.user_id = a.student_id
        WHERE u.role = 'student'
        GROUP BY u.user_id, u.name
        HAVING attendance_rate < 75
        ORDER BY attendance_rate ASC
        LIMIT 5
        """
    ).fetchall()
    
    # Get pending attendance requests
    pending_requests_list = conn.execute(
        """
        SELECT ar.*, u.name as student_name
        FROM attendance_requests ar
        JOIN users u ON ar.student_id = u.user_id
        WHERE ar.status = 'pending'
        ORDER BY ar.id DESC
        LIMIT 5
        """
    ).fetchall()
    
    conn.close()
    
    return render_template('teacher/dashboard.html',
                          total_students=total_students,
                          pending_requests=pending_requests,
                          at_risk_students=len(at_risk_students_result),
                          at_risk_students_list=at_risk_students_result,
                          pending_requests_list=pending_requests_list,
                          total_classes_today=6,
                          total_classes_taught=total_classes_taught)

@app.route('/teacher/profile')
def teacher_profile():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    teacher = conn.execute("SELECT * FROM users WHERE user_id = ?", (session['user_id'],)).fetchone()
    
    # Get teaching statistics
    total_students_result = conn.execute(
        "SELECT COUNT(*) as count FROM users WHERE role = 'student'"
    ).fetchone()
    total_students = total_students_result['count'] if total_students_result else 0
    
    total_classes_taught_result = conn.execute(
        "SELECT COUNT(*) as count FROM attendance WHERE marked_by = ?",
        (session['user_id'],)
    ).fetchone()
    total_classes_taught = total_classes_taught_result['count'] if total_classes_taught_result else 0
    
    # Calculate average attendance rate for classes taught by this teacher
    avg_attendance_result = conn.execute(
        """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) as present
        FROM attendance 
        WHERE marked_by = ?
        """, (session['user_id'],)
    ).fetchone()
    
    total_attended = avg_attendance_result['total'] if avg_attendance_result['total'] > 0 else 1
    present_count = avg_attendance_result['present'] or 0
    avg_attendance_rate = round((present_count / total_attended) * 100, 2) if total_attended > 0 else 0
    
    # Get feedback score (simulated - in real app this would be from evaluations)
    feedback_score = round(4.2 + (hash(session['user_id']) % 10) / 10, 1)  # Random score for demo
    
    conn.close()
    
    return render_template('teacher/profile.html', 
                          teacher=teacher,
                          total_students=total_students,
                          total_classes_taught=total_classes_taught,
                          avg_attendance_rate=avg_attendance_rate,
                          feedback_score=feedback_score)

@app.route('/teacher/update_profile', methods=['POST'])
def teacher_update_profile():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    phone = request.form.get('phone')
    address = request.form.get('address')
    department = request.form.get('department')
    designation = request.form.get('designation')
    
    # Phone number validation
    if phone and phone.strip():
        # Check if phone number is valid (10 digits)
        import re
        phone_pattern = r'^[0-9]{10}$'
        if not re.match(phone_pattern, phone.strip()):
            flash('Phone number must be exactly 10 digits.', 'error')
            return redirect(url_for('teacher_profile'))
    
    conn = get_db_connection()
    conn.execute(
        "UPDATE users SET department = ?, phone = ?, address = ?, designation = ? WHERE user_id = ?",
        (department, phone, address, designation, session['user_id'])
    )
    conn.commit()
    conn.close()
    
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('teacher_profile'))

@app.route('/api/get_students_by_dept_batch')
def api_get_students_by_dept_batch():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    department = request.args.get('department')
    batch = request.args.get('batch')
    
    if not department or not batch:
        return jsonify({'error': 'Department and batch are required'}), 400
    
    conn = get_db_connection()
    students = conn.execute(
        "SELECT user_id, name, department, batch FROM users WHERE role = 'student' AND department = ? AND batch = ? ORDER BY name",
        (department, batch)
    ).fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    students_list = [dict(student) for student in students]
    return jsonify(students_list)

@app.route('/teacher/mark_attendance', methods=['GET', 'POST'])
def teacher_mark_attendance():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Process attendance marking
        date = request.form['date']
        period = request.form['period']
        department = request.form['department']
        batch = request.form['batch']
        
        conn = get_db_connection()
        
        # Auto-lock old attendance records
        auto_lock_old_attendance(conn)
        
        # Get students in the specified department and batch
        students = conn.execute(
            "SELECT * FROM users WHERE role = 'student' AND department = ? AND batch = ?",
            (department, batch)
        ).fetchall()
        
        # Valid status options
        valid_statuses = get_attendance_status_options()
        marked_count = 0
        skipped_count = 0
        
        # Process attendance for each student
        for student in students:
            student_id = student['user_id']
            status = request.form.get(f'status_{student_id}')
            
            if status in valid_statuses:
                # Check if attendance already exists for this student, date, and period
                existing_attendance = conn.execute(
                    "SELECT * FROM attendance WHERE student_id = ? AND date = ? AND period = ?",
                    (student_id, date, period)
                ).fetchone()
                
                if existing_attendance:
                    # Check if record can be edited
                    can_edit, reason = check_attendance_editable(existing_attendance, session['role'])
                    
                    if can_edit:
                        old_status = existing_attendance['status']
                        # Update existing attendance
                        conn.execute(
                            """UPDATE attendance 
                               SET status = ?, marked_by = ? 
                               WHERE id = ?""",
                            (status, session['user_id'], existing_attendance['id'])
                        )
                        # Log the change
                        log_attendance_change(
                            conn, existing_attendance['id'], 'update',
                            old_status, status, session['user_id'], 'teacher'
                        )
                        marked_count += 1
                    else:
                        skipped_count += 1
                else:
                    # Insert new attendance record
                    cursor = conn.execute(
                        """INSERT INTO attendance 
                           (student_id, date, period, status, marked_by) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (student_id, date, period, status, session['user_id'])
                    )
                    # Log the creation
                    log_attendance_change(
                        conn, cursor.lastrowid, 'create',
                        None, status, session['user_id'], 'teacher'
                    )
                    marked_count += 1
        
        conn.commit()
        conn.close()
        
        if skipped_count > 0:
            flash(f'Attendance marked for {marked_count} students. {skipped_count} records were locked and skipped.', 'warning')
        else:
            flash(f'Attendance marked successfully for {marked_count} students!', 'success')
        return redirect(url_for('teacher_mark_attendance'))
    
    conn = get_db_connection()
    # Auto-lock old attendance records on page load
    auto_lock_old_attendance(conn)
    
    # Get departments and batches for dropdowns
    departments = conn.execute("SELECT DISTINCT department FROM users WHERE role = 'student' AND department IS NOT NULL").fetchall()
    batches = conn.execute("SELECT DISTINCT batch FROM users WHERE role = 'student' AND batch IS NOT NULL").fetchall()
    conn.close()
    
    return render_template('teacher/mark_attendance.html', 
                         departments=[dept['department'] for dept in departments],
                         batches=[batch['batch'] for batch in batches],
                         status_options=get_attendance_status_options())

@app.route('/teacher/daily_summary')
def teacher_daily_summary():
    """Teacher's end-of-day summary dashboard"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    teacher_id = session['user_id']
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Classes conducted today (unique department-batch-period combinations)
    classes_today = conn.execute('''
        SELECT COUNT(DISTINCT period || '-' || student_id) as count
        FROM attendance 
        WHERE marked_by = ? AND date = ?
    ''', (teacher_id, today)).fetchone()['count']
    
    # Attendance breakdown by period (1-6)
    hourly_breakdown = []
    for period in range(1, 7):
        stats = conn.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) as present,
                SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) as late,
                SUM(CASE WHEN status = 'excused' THEN 1 ELSE 0 END) as excused,
                SUM(CASE WHEN status = 'absent' THEN 1 ELSE 0 END) as absent
            FROM attendance 
            WHERE marked_by = ? AND date = ? AND period = ?
        ''', (teacher_id, today, period)).fetchone()
        
        hourly_breakdown.append({
            'period': period,
            'total': stats['total'] or 0,
            'present': stats['present'] or 0,
            'late': stats['late'] or 0,
            'excused': stats['excused'] or 0,
            'absent': stats['absent'] or 0
        })
    
    # Today's totals
    totals = conn.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) as present,
            SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) as late,
            SUM(CASE WHEN status = 'excused' THEN 1 ELSE 0 END) as excused,
            SUM(CASE WHEN status = 'absent' THEN 1 ELSE 0 END) as absent
        FROM attendance 
        WHERE marked_by = ? AND date = ?
    ''', (teacher_id, today)).fetchone()
    
    # Pending correction requests
    pending_requests = conn.execute('''
        SELECT ar.*, u.name as student_name
        FROM attendance_requests ar
        JOIN users u ON ar.student_id = u.user_id
        WHERE ar.status = 'pending'
        ORDER BY ar.id DESC
    ''').fetchall()
    
    # At-risk students (< 75% attendance)
    at_risk_students = conn.execute('''
        SELECT u.user_id, u.name, u.department, u.batch,
               COUNT(a.id) as total_records,
               SUM(CASE WHEN a.status = 'present' OR a.status = 'late' THEN 1 ELSE 0 END) as attended,
               ROUND(SUM(CASE WHEN a.status = 'present' OR a.status = 'late' THEN 1 ELSE 0 END) * 100.0 / COUNT(a.id), 1) as attendance_rate
        FROM users u
        JOIN attendance a ON u.user_id = a.student_id
        WHERE u.role = 'student'
        GROUP BY u.user_id
        HAVING attendance_rate < 75
        ORDER BY attendance_rate ASC
        LIMIT 10
    ''').fetchall()
    
    # Flagged entries needing review
    flagged_count = conn.execute('''
        SELECT COUNT(*) as count FROM attendance 
        WHERE needs_review = 1 AND marked_by = ?
    ''', (teacher_id,)).fetchone()['count']
    
    conn.close()
    
    return render_template('teacher/daily_summary.html',
                         today=today,
                         classes_today=classes_today,
                         hourly_breakdown=hourly_breakdown,
                         totals=totals,
                         pending_requests=pending_requests,
                         at_risk_students=at_risk_students,
                         flagged_count=flagged_count)

@app.route('/teacher/review_queue')
def teacher_review_queue():
    """Review attendance entries flagged for manual verification"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get all flagged attendance entries
    flagged_entries = conn.execute('''
        SELECT a.*, u.name as student_name, u.department, u.batch
        FROM attendance a
        JOIN users u ON a.student_id = u.user_id
        WHERE a.needs_review = 1
        ORDER BY a.date DESC, a.period ASC
    ''').fetchall()
    
    conn.close()
    
    return render_template('teacher/review_queue.html', flagged_entries=flagged_entries)

@app.route('/teacher/confirm_attendance/<int:attendance_id>', methods=['POST'])
def teacher_confirm_attendance(attendance_id):
    """Confirm or reject a flagged attendance entry"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    action = request.form.get('action')  # 'confirm' or 'reject'
    new_status = request.form.get('status', 'present')
    
    conn = get_db_connection()
    
    # Get the attendance record
    record = conn.execute('SELECT * FROM attendance WHERE id = ?', (attendance_id,)).fetchone()
    
    if not record:
        conn.close()
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    
    old_status = record['status']
    
    if action == 'confirm':
        # Mark as confirmed (remove needs_review flag)
        conn.execute('''
            UPDATE attendance 
            SET needs_review = 0
            WHERE id = ?
        ''', (attendance_id,))
        log_attendance_change(conn, attendance_id, 'confirm_review', old_status, old_status, 
                            session['user_id'], 'teacher', 'Manual verification confirmed')
    elif action == 'reject':
        # Update status and mark as reviewed
        conn.execute('''
            UPDATE attendance 
            SET status = ?, needs_review = 0
            WHERE id = ?
        ''', (new_status, attendance_id))
        log_attendance_change(conn, attendance_id, 'reject_review', old_status, new_status,
                            session['user_id'], 'teacher', 'Manual verification - status changed')
    
    conn.commit()
    conn.close()
    
    flash('Attendance record reviewed successfully!', 'success')
    return redirect(url_for('teacher_review_queue'))



@app.route('/api/class_monitor/frame_analysis', methods=['POST'])
def analyze_frame_for_class_monitor():
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        # Get image data from request
        image_data = request.json.get('image_data', '')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'}), 400
        
        # Decode the base64 image data
        import base64
        import numpy as np
        import cv2
        import time
        import random
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array and then to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        
        # Use LBPH recognizer if available (more accurate)
        # Use LBPH recognizer if available (more accurate)
        confidences = []
        if USE_LBPH:
            print("[DEBUG] Using LBPH Recognizer")
            try:
                recognized_names, face_locations, confidences = lbph_recognizer.recognize_face(frame)
            except ValueError:
                # Handle case where old version might be cached or error
                result = lbph_recognizer.recognize_face(frame)
                recognized_names, face_locations = result[0], result[1]
                confidences = [0.0] * len(recognized_names)
        else:
            # Fallback to old method
            print("[DEBUG] Using old face_recognizer (fallback)")
            face_recognizer.load_encodings()
            recognized_names, face_locations = face_recognizer.recognize_face(frame)
            confidences = [0.0] * len(recognized_names)
        
        # Debug: Log recognition results
        print(f"[DEBUG] Recognized names: {recognized_names}")
        print(f"[DEBUG] Face locations: {face_locations}")
        print(f"[DEBUG] Type of recognized_names: {type(recognized_names)}")
        
        # Get the registered students from the database
        conn = get_db_connection()
        registered_students_data = conn.execute(
            "SELECT user_id, name, department, batch FROM users WHERE role = 'student'"
        ).fetchall()
        conn.close()
        
        # Create a dictionary for quick lookup
        students_dict = {student['user_id']: dict(student) for student in registered_students_data}
        print(f"[DEBUG] Students in DB: {list(students_dict.keys())}")
        
        # Get filter parameters key
        filter_dept = request.json.get('department')
        filter_batch = request.json.get('batch')
        
        # Process recognized faces and match them to students
        detected_students = []
        timestamp = time.strftime('%H:%M:%S')
        
        # Helper to check if student matches filter
        def matches_filter(student_data):
            if filter_dept and student_data.get('department') != filter_dept:
                return False
            if filter_batch and student_data.get('batch') != filter_batch:
                return False
            return True
        
        if isinstance(recognized_names, list):
            # Multiple faces detected
            for i, name in enumerate(recognized_names):
                if name != "Unknown" and name in students_dict:
                    student_dict = students_dict[name].copy()
                    
                    # STRICT FILTERING: Ignore if not from selected class
                    if not matches_filter(student_dict):
                        print(f"[FILTERED] Ignoring {name} (Dept: {student_dict.get('department')}, Batch: {student_dict.get('batch')})")
                        continue
                    
                    # Generate behavior metrics
                    # Use actual confidence if available (LBPH returns distance, so we invert/normalize it roughly)
                    # LBPH distance: 0 is perfect, 100 is threshold.
                    # We want 0-100% confidence where 100 is perfect.
                if not isinstance(confidences, list):
                    # Fallback if confidences is not a list (e.g. older return format)
                    confidences = [0.0] * len(recognized_names)

                if name != "Unknown" and name in students_dict:
                    student_dict = students_dict[name].copy()
                    
                    # Generate behavior metrics based on REAL confidence
                    if i < len(confidences) and USE_LBPH:
                        lbph_dist = confidences[i]
                        # LBPH Mapping:
                        # Dist 0   -> 100% confidence
                        # Dist 50  -> 90% confidence (good match)
                        # Dist 80  -> 60% confidence (threshold)
                        # Dist 100 -> 0% confidence
                        
                        if lbph_dist < 50:
                            calc_conf = 90 + (50 - lbph_dist) * 0.2  # 90-100 range
                        elif lbph_dist < 80:
                            calc_conf = 60 + (80 - lbph_dist)  # 60-90 range
                        else:
                            calc_conf = max(0, 60 - (lbph_dist - 80) * 2) # Drop off quickly
                            
                        student_dict['confidence'] = round(calc_conf, 1)
                        student_dict['raw_confidence'] = round(lbph_dist, 1)
                    else:
                        # Fallback for non-LBPH (should use face_recognition distance if available)
                        # For now, give a reasonable "high confidence" since face_recognition is strict
                        # Only random part is slight jitter for realism in UI updates
                        student_dict['confidence'] = 92.5 
                        student_dict['raw_confidence'] = 0.0
                    
                    focus_level = random.choice(['Excellent', 'Very High', 'High', 'Good'])
                    engagement_level = random.choice(['Outstanding', 'Excellent', 'Very High', 'High'])
                    
                    student_dict['focus_level'] = focus_level
                    student_dict['engagement_level'] = engagement_level
                    student_dict['behavior_score'] = int(student_dict['confidence']) # Link behavior to detection confidence
                    detected_students.append(student_dict)
                    
                    print(f"[{timestamp}] FACE RECOGNIZED: {student_dict['name']} - Conf: {student_dict['confidence']}%")
        elif isinstance(recognized_names, str):
            # Single face detected (Legacy format handling)
            name = recognized_names
            if name != "Unknown" and name in students_dict:
                student_dict = students_dict[name].copy()
                
                # STRICT FILTERING: Ignore if not from selected class
                if not matches_filter(student_dict):
                    print(f"[FILTERED] Ignoring {name}")
                    return jsonify({'success': True, 'detected_students': []})
                
                # Assume high confidence for single match in legacy mode
                student_dict['confidence'] = 90.0
                student_dict['focus_level'] = "High"
                student_dict['engagement_level'] = "High"
                student_dict['behavior_score'] = 90
                detected_students.append(student_dict)
                
                print(f"[{timestamp}] FACE RECOGNIZED: {student_dict['name']} ({student_dict['user_id']}) - Legacy Mode")
        
        # Log detection summary
        if len(detected_students) > 0:
            print(f"📊 REAL-TIME FRAME ANALYSIS: {len(face_locations)} faces detected - {len(detected_students)} student(s) identified")
        else:
            print(f"[{timestamp}] NO STUDENTS RECOGNIZED - {len(face_locations)} faces detected but none matched")
        
        return jsonify({
            'success': True,
            'detected_students': detected_students,
            'count': len(detected_students),
            'faces_detected': len(face_locations),
            'message': f'Real-time face recognition complete - {len(detected_students)} student(s) identified'
        })
        
    except Exception as e:
        import traceback
        print(f"Error in real-time frame analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Error processing frame: {str(e)}'}), 500

@app.route('/api/save_monitoring_session', methods=['POST'])
def save_monitoring_session():
    """Save monitoring session data to database"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        data = request.json
        teacher_id = data.get('teacher_id')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        duration = data.get('duration')
        detected_students = data.get('detected_students', [])
        total_students = data.get('total_students', 0)
        monitoring_data = data.get('data', [])
        
        # Convert lists to JSON strings for storage
        import json
        detected_students_json = json.dumps(detected_students)
        monitoring_data_json = json.dumps(monitoring_data)
        
        conn = get_db_connection()
        conn.execute(
            """INSERT INTO monitoring_sessions 
               (teacher_id, start_time, end_time, duration, detected_students, total_students, monitoring_data)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (teacher_id, start_time, end_time, duration, detected_students_json, total_students, monitoring_data_json)
        )
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Monitoring session saved successfully'})
        
    except Exception as e:
        print(f"Error saving monitoring session: {str(e)}")
        return jsonify({'success': False, 'message': f'Error saving session: {str(e)}'}), 500

@app.route('/api/create_incident_report', methods=['POST'])
def create_incident_report():
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        data = request.json
        student_id = data.get('student_id')
        timestamp = data.get('timestamp')
        flagged_by = data.get('flagged_by')
        reason = data.get('reason', 'Behavior concern reported')
        severity = data.get('severity', 'medium')
        
        if not student_id:
            return jsonify({'success': False, 'message': 'Student ID is required'}), 400
        
        # Save incident report to database
        conn = get_db_connection()
        conn.execute(
            """INSERT INTO incident_reports (student_id, flagged_by, reason, severity, timestamp) 
               VALUES (?, ?, ?, ?, ?)""",
            (student_id, flagged_by, reason, severity, timestamp)
        )
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Incident report created successfully'})
        
    except Exception as e:
        print(f"Error creating incident report: {str(e)}")
        return jsonify({'success': False, 'message': f'Error creating incident report: {str(e)}'}), 500

@app.route('/student/dashboard')
def student_dashboard():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    
    try:
        conn = get_db_connection()
        
        student_id = session['user_id']
        
        # Get student's attendance data with specific counts (Periods)
        # Leave status counts as 0.5 attendance credit for percentage
        attendance_stats = conn.execute('''
            SELECT COUNT(*) as total_classes, 
                   SUM(CASE WHEN status = 'present' THEN 1 
                            WHEN status = 'leave' THEN 0.5 
                            ELSE 0 END) as attended_credit
            FROM attendance 
            WHERE student_id = ?
        ''', (student_id,)).fetchone()
        
        # Get statistics by DAY (distinct dates) for dashboard cards
        # This prevents "23 leaves" when it was just 4 days
        day_stats = conn.execute('''
            SELECT 
                COUNT(DISTINCT CASE WHEN status = 'present' THEN date END) as present_days,
                COUNT(DISTINCT CASE WHEN status = 'absent' THEN date END) as absent_days,
                COUNT(DISTINCT CASE WHEN status = 'late' THEN date END) as late_days,
                COUNT(DISTINCT CASE WHEN status = 'leave' THEN date END) as leave_days
            FROM attendance 
            WHERE student_id = ?
        ''', (student_id,)).fetchone()
        
        total_classes = attendance_stats['total_classes'] if attendance_stats['total_classes'] > 0 else 1
        attended_credit = attendance_stats['attended_credit'] or 0
        attendance_percentage = round((attended_credit / total_classes) * 100, 2) if total_classes > 0 else 0
        
        # Prepare real stats for dashboard (using DAYS)
        stats = {
            'present': day_stats['present_days'] or 0,
            'absent': day_stats['absent_days'] or 0,
            'late': day_stats['late_days'] or 0,
            'leave': day_stats['leave_days'] or 0,
            'total_classes': attendance_stats['total_classes'] or 0
        }
        
        # Get attendance by subject (department)
        subject_attendance = conn.execute(
            """
            SELECT u.department as subject, 
                   COUNT(*) as total_classes,
                   SUM(CASE WHEN a.status = 'present' THEN 1 ELSE 0 END) as attended
            FROM attendance a
            JOIN users u ON a.marked_by = u.user_id
            WHERE a.student_id = ?
            GROUP BY u.department
            """, (student_id,)
        ).fetchall()
        
        # Get recent attendance records
        recent_attendance = conn.execute(
            "SELECT * FROM attendance WHERE student_id = ? ORDER BY date DESC LIMIT 10",
            (student_id,)
        ).fetchall()
        
        # Get any pending attendance requests for this student
        pending_request = conn.execute(
            "SELECT * FROM attendance_requests WHERE student_id = ? AND status = 'pending' ORDER BY id DESC LIMIT 1",
            (student_id,)
        ).fetchone()
        
        # Get student information
        student_info = conn.execute("SELECT * FROM users WHERE user_id = ?", (student_id,)).fetchone()
        conn.close()
        
        return render_template('student/dashboard.html',
                              user=dict(student_info),
                              attendance_percentage=attendance_percentage,
                              total_classes=total_classes,
                              subject_attendance=subject_attendance,
                              recent_attendance=recent_attendance,
                              pending_request=pending_request,
                              stats=stats)
    except Exception as e:
        print(f"Error in student dashboard: {str(e)}")
        flash('An error occurred while loading your dashboard. Please try again.', 'error')
        return redirect(url_for('login'))

@app.route('/student/attendance_history')
def student_attendance_history():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    student_id = session['user_id']
    
    # Get filters from request
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Build query
    query = """
        SELECT a.*, u.name as teacher_name
        FROM attendance a
        JOIN users u ON a.marked_by = u.user_id
        WHERE a.student_id = ?
    """
    params = [student_id]
    
    if start_date:
        query += " AND a.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND a.date <= ?"
        params.append(end_date)
        
    query += " ORDER BY a.date DESC, a.period DESC"
    
    # Get student's attendance data
    attendance_data = conn.execute(query, tuple(params)).fetchall()
    
    # Get student info
    student_info = conn.execute("SELECT * FROM users WHERE user_id = ?", (student_id,)).fetchone()
    
    conn.close()
    
    return render_template('student/attendance_history.html', 
                          attendance_data=attendance_data,
                          student_info=dict(student_info),
                          start_date=start_date,
                          end_date=end_date)

@app.route('/student/request_attendance', methods=['POST'])
def student_request_attendance():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    
    try:
        date = request.form['date']
        period = request.form['period']
        reason = request.form['reason']
        
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO attendance_requests (student_id, date, period, reason, status) VALUES (?, ?, ?, ?, 'pending')",
            (session['user_id'], date, period, reason)
        )
        conn.commit()
        conn.close()
        
        flash('Attendance request submitted successfully!', 'success')
        return redirect(url_for('student_dashboard'))
    except Exception as e:
        print(f"Error submitting attendance request: {str(e)}")
        flash('An error occurred while submitting your request. Please try again.', 'error')
        return redirect(url_for('student_dashboard'))


@app.route('/teacher/attendance_requests')
def teacher_attendance_requests():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get pending attendance requests
    requests = conn.execute("""
        SELECT ar.*, u.name as student_name, u.user_id as student_user_id
        FROM attendance_requests ar
        JOIN users u ON ar.student_id = u.user_id
        WHERE ar.status = 'pending'
        ORDER BY ar.id DESC
    """).fetchall()
    
    conn.close()
    
    return render_template('teacher/attendance_requests.html', requests=requests)


@app.route('/teacher/approve_request/<int:request_id>', methods=['POST'])
def teacher_approve_request(request_id):
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        conn = get_db_connection()
        
        # Get the request details
        request_details = conn.execute(
            "SELECT student_id, date, period FROM attendance_requests WHERE id = ? AND status = 'pending'",
            (request_id,)
        ).fetchone()
        
        if not request_details:
            conn.close()
            return jsonify({'success': False, 'message': 'Request not found or already processed'})
        
        # Update the attendance record to 'present' (or whatever was requested)
        conn.execute(
            "UPDATE attendance SET status = 'present' WHERE student_id = ? AND date = ? AND period = ?",
            (request_details['student_id'], request_details['date'], request_details['period'])
        )
        
        # Update the request status to 'approved'
        conn.execute(
            "UPDATE attendance_requests SET status = 'approved' WHERE id = ?",
            (request_id,)
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Request approved successfully'})
    except Exception as e:
        # Log the error for debugging
        print(f"Error approving request: {str(e)}")
        return jsonify({'success': False, 'message': f'Error processing request: {str(e)}'}), 500


@app.route('/teacher/reject_request/<int:request_id>', methods=['POST'])
def teacher_reject_request(request_id):
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        conn = get_db_connection()
        
        # Get the request details
        request_details = conn.execute(
            "SELECT student_id, date, period FROM attendance_requests WHERE id = ? AND status = 'pending'",
            (request_id,)
        ).fetchone()
        
        if not request_details:
            conn.close()
            return jsonify({'success': False, 'message': 'Request not found or already processed'})
        
        # Update the request status to 'rejected'
        conn.execute(
            "UPDATE attendance_requests SET status = 'rejected' WHERE id = ?",
            (request_id,)
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Request rejected successfully'})
    except Exception as e:
        # Log the error for debugging
        print(f"Error rejecting request: {str(e)}")
        return jsonify({'success': False, 'message': f'Error processing request: {str(e)}'}), 500


@app.route('/teacher/student_details/<string:student_id>')
def teacher_student_details(student_id):
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        conn = get_db_connection()
        
        # Get student details
        student = conn.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (student_id,)
        ).fetchone()
        
        conn.close()
        
        if student:
            s_dict = dict(student)
            
            # Ensure face image path is available
            if not s_dict.get('face_image_path'):
                 path = get_user_face_image_path(s_dict['user_id'], s_dict.get('role', 'student'))
                 if path:
                     s_dict['face_image_path'] = path
            
            # Construct accessible URL
            if s_dict.get('face_image_path'):
                role = s_dict.get('role', 'student')
                folder_map = {'student': 'students', 'teacher': 'teachers', 'admin': 'admins'}
                folder = folder_map.get(role, 'others')
                # Path stored in DB is relative to folder (e.g. student001/img.jpg)
                # DB path strips it. So we must add it back.
                s_dict['face_image_url'] = f"/dataset/{folder}/{s_dict['face_image_path']}"
            else:
                 s_dict['face_image_url'] = None
                 
            return jsonify({
                'success': True,
                'student': s_dict
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Student not found'
            }), 404
    except Exception as e:
        print(f"Error getting student details: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error retrieving student details: {str(e)}'
        }), 500


@app.route('/admin/attendance_requests')
def admin_attendance_requests():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get all attendance requests (not just pending ones) so admin can see all
    requests = conn.execute("""
        SELECT ar.*, u.name as student_name, u.user_id as student_user_id
        FROM attendance_requests ar
        JOIN users u ON ar.student_id = u.user_id
        ORDER BY ar.id DESC
    """).fetchall()
    
    conn.close()
    
    return render_template('admin/attendance_requests.html', requests=requests)

@app.route('/admin/override_attendance/<int:attendance_id>', methods=['POST'])
def admin_override_attendance(attendance_id):
    """Admin can override locked attendance records"""
    if 'user_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    new_status = request.form.get('status')
    justification = request.form.get('justification', '')
    
    if not new_status:
        return jsonify({'success': False, 'error': 'Status is required'}), 400
    
    if not justification:
        return jsonify({'success': False, 'error': 'Justification is required for admin override'}), 400
    
    valid_statuses = get_attendance_status_options()
    if new_status not in valid_statuses:
        return jsonify({'success': False, 'error': f'Invalid status. Must be one of: {valid_statuses}'}), 400
    
    conn = get_db_connection()
    
    # Get the attendance record
    record = conn.execute('SELECT * FROM attendance WHERE id = ?', (attendance_id,)).fetchone()
    
    if not record:
        conn.close()
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    
    old_status = record['status']
    was_locked = record['is_locked']
    
    # Update the record (admin can edit even locked records)
    conn.execute('''
        UPDATE attendance 
        SET status = ?
        WHERE id = ?
    ''', (new_status, attendance_id))
    
    # Log the admin override
    action = 'admin_override' if was_locked else 'admin_update'
    log_attendance_change(conn, attendance_id, action, old_status, new_status,
                         session['user_id'], 'admin', justification)
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': f'Attendance updated from {old_status} to {new_status}'})

@app.route('/admin/attendance_audit/<int:attendance_id>')
def admin_attendance_audit(attendance_id):
    """Get audit history for an attendance record"""
    if 'user_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    conn = get_db_connection()
    
    # Get the attendance record
    record = conn.execute('''
        SELECT a.*, u.name as student_name
        FROM attendance a
        JOIN users u ON a.student_id = u.user_id
        WHERE a.id = ?
    ''', (attendance_id,)).fetchone()
    
    if not record:
        conn.close()
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    
    # Get audit log
    audit_log = conn.execute('''
        SELECT al.*, ed.name as editor_name
        FROM attendance_audit_log al
        LEFT JOIN users ed ON al.changed_by = ed.user_id
        WHERE al.attendance_id = ?
        ORDER BY al.timestamp DESC
    ''', (attendance_id,)).fetchall()
    
    conn.close()
    
    return jsonify({
        'success': True,
        'record': dict(record),
        'audit_log': [dict(log) for log in audit_log]
    })
@app.route('/student/my_requests')
def student_view_requests():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get attendance requests for the logged-in student
    requests = conn.execute("""
        SELECT *
        FROM attendance_requests
        WHERE student_id = ?
        ORDER BY id DESC
    """, (session['user_id'],)).fetchall()
    
    conn.close()
    
    return render_template('student/my_requests.html', requests=requests)

# ============================================
# Leave Request Feature - Students can request leave,
# approved leaves count as 0.5 attendance credit
# ============================================

@app.route('/student/leave_requests', methods=['GET', 'POST'])
def student_leave_requests():
    if 'user_id' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    if request.method == 'POST':
        # Submit new leave request
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        reason = request.form['reason']
        leave_type = request.form.get('leave_type', 'personal')
        teacher_id = request.form.get('teacher_id')  # Get selected teacher
        
        try:
            conn.execute('''
                INSERT INTO leave_requests (student_id, start_date, end_date, reason, leave_type, teacher_id, status)
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
            ''', (session['user_id'], start_date, end_date, reason, leave_type, teacher_id))
            conn.commit()
            flash('Leave request submitted successfully!', 'success')
        except Exception as e:
            print(f"Error submitting leave request: {e}")
            flash('Error submitting leave request. Please try again.', 'error')
        
        return redirect(url_for('student_leave_requests'))
    
    # GET: Fetch all leave requests for this student
    leave_requests = conn.execute('''
        SELECT lr.*, t.name as teacher_name 
        FROM leave_requests lr
        LEFT JOIN users t ON lr.teacher_id = t.user_id
        WHERE lr.student_id = ? 
        ORDER BY lr.created_at DESC
    ''', (session['user_id'],)).fetchall()
    
    # Fetch list of teachers for the dropdown
    teachers = conn.execute("SELECT user_id, name, department FROM users WHERE role = 'teacher' ORDER BY name").fetchall()
    
    conn.close()
    
    return render_template('student/leave_requests.html', leave_requests=leave_requests, teachers=teachers)




@app.route('/teacher/leave_requests')
def teacher_leave_requests():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    teacher_id = session['user_id']
    conn = get_db_connection()
    
    # Get pending leave requests assigned to this teacher OR fallback to previous logic (taught students)
    # Logic: 
    # 1. Matches teacher_id directly (Explicit assignment)
    # 2. OR teacher_id is NULL AND teacher has marked attendance for student (Implicit assignment fallback)
    leave_requests = conn.execute('''
        SELECT DISTINCT lr.*, u.name as student_name, u.department, u.batch
        FROM leave_requests lr
        JOIN users u ON lr.student_id = u.user_id
        WHERE lr.status = 'pending'
        AND (
            lr.teacher_id = ? 
            OR (
                lr.teacher_id IS NULL AND lr.student_id IN (
                    SELECT DISTINCT student_id FROM attendance WHERE marked_by = ?
                )
            )
        )
        ORDER BY lr.created_at DESC
    ''', (teacher_id, teacher_id)).fetchall()
    
    conn.close()
    
    return render_template('teacher/leave_requests.html', leave_requests=leave_requests)


@app.route('/teacher/approve_leave/<int:request_id>', methods=['POST'])
def teacher_approve_leave(request_id):
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    conn = None
    try:
        teacher_id = session['user_id']
        conn = get_db_connection()
        
        # Get leave request details
        leave_req = conn.execute('''
            SELECT * FROM leave_requests WHERE id = ? AND status = 'pending'
        ''', (request_id,)).fetchone()
        
        if not leave_req:
            return jsonify({'success': False, 'message': 'Leave request not found or already processed'})
        
        # Verify that this teacher has taught this student
        has_taught = conn.execute('''
            SELECT 1 FROM attendance 
            WHERE student_id = ? AND marked_by = ?
            LIMIT 1
        ''', (leave_req['student_id'], teacher_id)).fetchone()
        
        if not has_taught:
            return jsonify({'success': False, 'message': 'You can only approve leave for students you have taught'})
        
        # Update leave request status to approved
        conn.execute('''
            UPDATE leave_requests 
            SET status = 'approved', reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (teacher_id, request_id))
        
        # Create attendance records with 'leave' status for each day in the leave period
        # These will count as 0.5 attendance when calculating attendance percentage
        from datetime import datetime, timedelta
        
        start_date = datetime.strptime(leave_req['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(leave_req['end_date'], '%Y-%m-%d')
        
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
                
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Create leave attendance records for all 6 periods
            for period in range(1, 7):
                # Check if attendance record already exists
                existing = conn.execute('''
                    SELECT id FROM attendance 
                    WHERE student_id = ? AND date = ? AND period = ?
                ''', (leave_req['student_id'], date_str, period)).fetchone()
                
                if existing:
                    # Update existing record to 'leave'
                    conn.execute('''
                        UPDATE attendance 
                        SET status = 'leave'
                        WHERE id = ?
                    ''', (existing['id'],))
                else:
                    # Create new 'leave' record
                    conn.execute('''
                        INSERT INTO attendance (student_id, date, period, status, marked_by)
                        VALUES (?, ?, ?, 'leave', ?)
                    ''', (leave_req['student_id'], date_str, period, teacher_id))
            
            current_date += timedelta(days=1)
        
        conn.commit()
        return jsonify({'success': True, 'message': 'Leave approved successfully. Attendance marked as leave.'})
    except Exception as e:
        print(f"Error approving leave: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500
    finally:
        if conn:
            conn.close()


@app.route('/teacher/reject_leave/<int:request_id>', methods=['POST'])
def teacher_reject_leave(request_id):
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    conn = None
    try:
        teacher_id = session['user_id']
        conn = get_db_connection()
        
        # Get leave request
        leave_req = conn.execute('''
            SELECT * FROM leave_requests WHERE id = ? AND status = 'pending'
        ''', (request_id,)).fetchone()
        
        if not leave_req:
            return jsonify({'success': False, 'message': 'Leave request not found or already processed'})
        
        # Verify that this teacher has taught this student
        has_taught = conn.execute('''
            SELECT 1 FROM attendance 
            WHERE student_id = ? AND marked_by = ?
            LIMIT 1
        ''', (leave_req['student_id'], teacher_id)).fetchone()
        
        if not has_taught:
            return jsonify({'success': False, 'message': 'You can only reject leave for students you have taught'})
        
        remarks = request.form.get('remarks', '')
        
        # Update leave request status to rejected
        conn.execute('''
            UPDATE leave_requests 
            SET status = 'rejected', reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP, teacher_remarks = ?
            WHERE id = ?
        ''', (teacher_id, remarks, request_id))
        
        conn.commit()
        return jsonify({'success': True, 'message': 'Leave request rejected.'})
    except Exception as e:
        print(f"Error rejecting leave: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    report_type = request.form.get('report_type')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    conn = get_db_connection()
    
    context = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date_range': f"{start_date or 'N/A'} to {end_date or 'N/A'}" if start_date or end_date else "All Time"
    }

    try:
        if report_type == 'daily_attendance':
            target_date = start_date or datetime.now().strftime('%Y-%m-%d')
            context['title'] = f"Daily Attendance Report - {target_date}"
            context['columns'] = ['Student Name', 'Department', 'Batch', 'Period', 'Status', 'Teacher']
            
            data = conn.execute("""
                SELECT u.name, u.department, u.batch, a.period, a.status, t.name as teacher_name
                FROM attendance a
                JOIN users u ON a.student_id = u.user_id
                JOIN users t ON a.marked_by = t.user_id
                WHERE a.date = ?
                ORDER BY u.department, u.batch, a.period
            """, (target_date,)).fetchall()
            
            context['rows'] = [[d['name'], d['department'], d['batch'], f"Period {d['period']}", d['status'], d['teacher_name']] for d in data]

        elif report_type == 'weekly_summary':
             # Logic for last 7 days or range
            s_date = start_date or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            e_date = end_date or datetime.now().strftime('%Y-%m-%d')
            context['title'] = f"Weekly Attendance Summary ({s_date} to {e_date})"
            context['columns'] = ['Date', 'Total Present', 'Total Absent', 'Total Late']
            
            data = conn.execute("""
                SELECT date, 
                       SUM(CASE WHEN status='present' THEN 1 ELSE 0 END) as present,
                       SUM(CASE WHEN status='absent' THEN 1 ELSE 0 END) as absent,
                       SUM(CASE WHEN status='late' THEN 1 ELSE 0 END) as late
                FROM attendance
                WHERE date BETWEEN ? AND ?
                GROUP BY date
                ORDER BY date DESC
            """, (s_date, e_date)).fetchall()
            
            context['rows'] = [[d['date'], d['present'], d['absent'], d['late']] for d in data]

        elif report_type == 'monthly_analysis':
             s_date = start_date or (datetime.now().replace(day=1)).strftime('%Y-%m-%d') # Start of month
             context['title'] = "Monthly Attendance Analysis"
             context['columns'] = ['Student Name', 'Total Classes', 'Present', 'Attendance %']
             
             data = conn.execute("""
                SELECT u.name, COUNT(a.id) as total,
                       SUM(CASE WHEN a.status='present' THEN 1 ELSE 0 END) as present
                FROM attendance a
                JOIN users u ON a.student_id = u.user_id
                WHERE a.date >= ?
                GROUP BY u.user_id
             """, (s_date,)).fetchall()
             
             rows = []
             for d in data:
                 percentage = round((d['present'] / d['total'] * 100), 1) if d['total'] > 0 else 0
                 rows.append([d['name'], d['total'], d['present'], f"{percentage}%"])
             context['rows'] = rows

        elif report_type == 'student_performance':
            context['title'] = "Student Performance Report"
            context['columns'] = ['Student Name', 'ID', 'Department', 'Total Attendance']
            
            # Simple query for all students and their total attendance count
            data = conn.execute("""
                SELECT u.name, u.user_id, u.department, COUNT(a.id) as count
                FROM users u
                LEFT JOIN attendance a ON u.user_id = a.student_id AND a.status = 'present'
                WHERE u.role = 'student'
                GROUP BY u.user_id
                ORDER BY count DESC
            """).fetchall()
            
            context['rows'] = [[d['name'], d['user_id'], d['department'], d['count']] for d in data]
            
        elif report_type == 'teacher_activity':
            context['title'] = "Teacher Activity Report"
            context['columns'] = ['Teacher Name', 'Classes Conducted (Attendance Marked)']
            
            data = conn.execute("""
                SELECT u.name, COUNT(DISTINCT a.id) as count
                FROM users u
                LEFT JOIN attendance a ON u.user_id = a.marked_by
                WHERE u.role = 'teacher'
                GROUP BY u.user_id
                ORDER BY count DESC
            """).fetchall()
            
            context['rows'] = [[d['name'], d['count']] for d in data]

        else:
            flash('Invalid report type selected.', 'error')
            return redirect(url_for('reports'))

        conn.close()

        # Generate PDF
        html = render_template('report_pdf.html', **context)
        
        # Ensure reports directory exists
        reports_dir = os.path.join(app.root_path, 'static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        filename = f"report_{report_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        file_path = os.path.join(reports_dir, filename)
        
        with open(file_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(io.BytesIO(html.encode('utf-8')), dest=pdf_file)

        if pisa_status.err:
             print(f"ERROR: PDF generation failed. pisa_status.err={pisa_status.err}")
             flash('Error generating PDF report. Please try again or contact admin.', 'error')
             return redirect(url_for('reports'))
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )

    except Exception as e:
        print(f"Error generating report: {e}")
        # Only close if still open (though get_db_connection opens new one each time)
        # conn variable might not exist if exception happened before get_db_connection
        # but here it is defined early.
        try:
             conn.close()
        except:
             pass
             
        flash(f"Error generating report: {str(e)}", 'error')
        return redirect(url_for('reports'))

@app.route('/reports')
def reports():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get overall stats
    total_students = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'student'").fetchone()['count']
    total_teachers = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'teacher'").fetchone()['count']
    total_admins = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'admin'").fetchone()['count']
    
    # Get attendance stats
    total_attendance_records = conn.execute("SELECT COUNT(*) as count FROM attendance").fetchone()['count']
    
    # Get recent attendance
    recent_attendance = conn.execute("""
        SELECT u.name, u.user_id, a.date, a.period, a.status 
        FROM attendance a 
        JOIN users u ON a.student_id = u.user_id 
        ORDER BY a.id DESC LIMIT 10
    """).fetchall()
    
    # Get attendance by date
    attendance_by_date = conn.execute("""
        SELECT date, COUNT(*) as count 
        FROM attendance 
        GROUP BY date 
        ORDER BY date DESC 
        LIMIT 7
    """).fetchall()
    
    # Get attendance by teacher
    attendance_by_teacher = conn.execute("""
        SELECT u.name, COUNT(a.id) as count 
        FROM attendance a 
        JOIN users u ON a.marked_by = u.user_id 
        GROUP BY a.marked_by
    """).fetchall()
    
    conn.close()
    
    return render_template('reports.html',
                          total_students=total_students,
                          total_teachers=total_teachers,
                          total_admins=total_admins,
                          total_attendance_records=total_attendance_records,
                          recent_attendance=recent_attendance,
                          attendance_by_date=attendance_by_date,
                          attendance_by_teacher=attendance_by_teacher)

@app.route('/reports/daily', methods=['GET', 'POST'])
def reports_daily():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    if request.method == 'POST':
        date = request.form['date']
        
        # Get attendance data for the selected date
        attendance_data = conn.execute("""
            SELECT u.name as student_name, u.department, u.batch, a.period, a.status, t.name as teacher_name
            FROM attendance a
            JOIN users u ON a.student_id = u.user_id
            JOIN users t ON a.marked_by = t.user_id
            WHERE a.date = ?
            ORDER BY u.department, u.batch, a.period
        """, (date,)).fetchall()
        
        conn.close()
        
        return render_template('reports_daily.html', attendance_data=attendance_data, report_type="Daily Attendance Report")
    
    # For GET request, show form
    conn.close()
    return render_template('reports_daily.html', report_type="Daily Attendance Report")



@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    if request.path == '/register':
        flash('File too large. Please use a smaller image (under 32MB).', 'error')
        return redirect(url_for('register'))
    return jsonify({'error': 'File too large. Please use a smaller image (under 32MB).'}), 413

@app.route('/face_images/<path:filename>')
def face_images(filename):
    # Check student directory first
    student_path = os.path.join(app.root_path, 'dataset', 'students', filename)
    if os.path.exists(student_path):
        return send_from_directory(os.path.join(app.root_path, 'dataset', 'students'), filename)
    
    # Check teacher directory
    teacher_path = os.path.join(app.root_path, 'dataset', 'teachers', filename)
    if os.path.exists(teacher_path):
        return send_from_directory(os.path.join(app.root_path, 'dataset', 'teachers'), filename)
        
    return jsonify({'error': 'Image not found'}), 404

@app.route('/api/users/<user_id>')
def api_get_user_details(user_id):
    if 'user_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    
    if user:
        # Convert row to dict
        user_dict = dict(user)
        # Remove sensitive data
        if 'password' in user_dict:
            del user_dict['password']
        if 'password_hash' in user_dict:
            del user_dict['password_hash']
        return jsonify({'success': True, 'user': user_dict})
    
    return jsonify({'success': False, 'message': 'User not found'}), 404

@app.route('/teacher/class_monitor')
def teacher_class_monitor():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get filter parameters
    department = request.args.get('department')
    batch = request.args.get('batch')
    
    # Base query
    query = "SELECT user_id, name, department, batch FROM users WHERE role = 'student'"
    params = []
    
    if department:
        query += " AND department = ?"
        params.append(department)
        
    if batch:
        query += " AND batch = ?"
        params.append(batch)
        
    query += " ORDER BY name"
    
    # Get filtered students
    students = conn.execute(query, tuple(params)).fetchall()
    
    # Get all departments and batches for dropdowns
    departments = conn.execute("SELECT DISTINCT department FROM users WHERE role='student' AND department IS NOT NULL ORDER BY department").fetchall()
    batches = conn.execute("SELECT DISTINCT batch FROM users WHERE role='student' AND batch IS NOT NULL ORDER BY batch").fetchall()
    
    conn.close()
    
    return render_template('teacher/class_monitor.html',
                          students=students,
                          departments=[d['department'] for d in departments],
                          batches=[b['batch'] for b in batches],
                          selected_dept=department,
                          selected_batch=batch)

@app.route('/ai_tutor')
def ai_tutor_interface():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # AI Tutor is only available for teachers and students
    if session['role'] == 'admin':
        flash('AI Tutor is available only for teachers and students.', 'info')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('ai_tutor.html')

@app.route('/ai_tutor/chat', methods=['POST'], endpoint='ai_tutor_chat_endpoint')
def ai_tutor_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    user_message = data.get('message')
    user_role = session.get('role', 'student')
    session_id = data.get('session_id')  # Get session_id from request
    
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400
    
    # Get response from AI Tutor
    response = ai_tutor.chat(user_message, user_role)
    
    # Save to history
    try:
        import uuid
        conn = get_db_connection()
        
        # If no session_id provided, create new session
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Check if this is the first message in the session (for title generation)
        existing = conn.execute(
            'SELECT COUNT(*) FROM chat_history WHERE session_id = ?',
            (session_id,)
        ).fetchone()[0]
        
        # Generate session title from first message
        session_title = user_message[:50] + ('...' if len(user_message) > 50 else '') if existing == 0 else None
        
        # Insert message with session info
        if session_title:
            conn.execute('''
                INSERT INTO chat_history (user_id, user_role, message, response, session_id, session_title)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], user_role, user_message, response['answer'], session_id, session_title))
        else:
            # Get existing session title
            session_title = conn.execute(
                'SELECT session_title FROM chat_history WHERE session_id = ? LIMIT 1',
                (session_id,)
            ).fetchone()[0]
            conn.execute('''
                INSERT INTO chat_history (user_id, user_role, message, response, session_id, session_title)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], user_role, user_message, response['answer'], session_id, session_title))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat history: {e}")
        # Don't fail the request if history saving fails
    
    return jsonify({'response': response, 'session_id': session_id})

@app.route('/ai_tutor/history')
def ai_tutor_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        # Get limit and session_id from query parameters
        limit = request.args.get('limit', 50, type=int)
        session_id = request.args.get('session_id', None)
        # Cap at 200 to prevent performance issues
        limit = min(limit, 200)
        
        conn = get_db_connection()
        
        if session_id:
            # Filter by session_id
            history = conn.execute('''
                SELECT message, response, timestamp 
                FROM chat_history 
                WHERE user_id = ? AND session_id = ?
                ORDER BY timestamp ASC 
                LIMIT ?
            ''', (session['user_id'], session_id, limit)).fetchall()
        else:
            # Return all history
            history = conn.execute('''
                SELECT message, response, timestamp 
                FROM chat_history 
                WHERE user_id = ? 
                ORDER BY timestamp ASC 
                LIMIT ?
            ''', (session['user_id'], limit)).fetchall()
        
        conn.close()
        
        return jsonify({
            'success': True, 
            'history': [dict(row) for row in history]
        })
    except Exception as e:
        print(f"Error fetching filtered history: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/ai_tutor/sessions')
def ai_tutor_sessions():
    """Get list of all chat sessions for current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        conn = get_db_connection()
        sessions = conn.execute('''
            SELECT 
                session_id,
                session_title,
                MAX(timestamp) as last_message_time,
                COUNT(*) as message_count
            FROM chat_history
            WHERE user_id = ?
            GROUP BY session_id
            ORDER BY last_message_time DESC
        ''', (session['user_id'],)).fetchall()
        conn.close()
        
        return jsonify({
            'success': True,
            'sessions': [dict(s) for s in sessions]
        })
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/ai_tutor/session/new', methods=['POST'])
def create_new_session():
    """Create a new chat session"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    import uuid
    new_session_id = str(uuid.uuid4())
    
    return jsonify({
        'success': True,
        'session_id': new_session_id
    })

@app.route('/ai_tutor/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        conn = get_db_connection()
        conn.execute('''
            DELETE FROM chat_history 
            WHERE user_id = ? AND session_id = ?
        ''', (session['user_id'], session_id))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting session: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/users/<user_id>/reset_password', methods=['POST'])
def api_reset_password(user_id):
    if 'user_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    data = request.json
    new_password = data.get('password')
    
    if not new_password:
        return jsonify({'success': False, 'message': 'Password is required'}), 400
        
    password_hash = hashlib.sha256(new_password.encode()).hexdigest()
    
    try:
        conn = get_db_connection()
        # Verify user exists
        user = conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not user:
            conn.close()
            return jsonify({'success': False, 'message': 'User not found'}), 404
            
        conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (password_hash, user_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Password updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/api/admin/dashboard_stats')
def api_admin_dashboard_stats():
    if 'user_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    conn = get_db_connection()
    
    # Get counts (Registered/Active users only)
    total_students = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'student' AND is_active = 1").fetchone()['count']
    total_teachers = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'teacher' AND is_active = 1").fetchone()['count']
    total_admins = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'admin'").fetchone()['count']
    
    # Get attendance rate
    # (Attendance records are naturally linked to student_id which must exist, but we still use active count)
    attendance_stats = conn.execute("""
        SELECT 
            COUNT(*) as total_records,
            SUM(CASE WHEN status = 'present' THEN 1 
                     WHEN status = 'leave' THEN 0.5 
                     ELSE 0 END) as attended_credit
        FROM attendance
    """).fetchone()
    
    overall_attendance_rate = 0
    if attendance_stats['total_records'] > 0:
        overall_attendance_rate = round((attendance_stats['attended_credit'] / attendance_stats['total_records']) * 100, 1)
    
    low_attendance_students = 0
    if total_students > 0:
        high_performers_count = conn.execute("""
            SELECT COUNT(*) as count FROM (
                SELECT student_id
                FROM attendance
                GROUP BY student_id
                HAVING (SUM(CASE WHEN status = 'present' THEN 1 
                                 WHEN status = 'leave' THEN 0.5 
                                 ELSE 0 END) * 100.0 / COUNT(*)) >= 85
            ) AS high_perf
        """).fetchone()['count']
        low_attendance_students = total_students - high_performers_count
    
    # Pending requests
    pending_requests = conn.execute("SELECT COUNT(*) as count FROM attendance_requests WHERE status = 'pending'").fetchone()['count']
    
    # Recent activities (simplify and filter for active users only)
    recent_users_rows = conn.execute("""
        SELECT name, user_id, role, department, created_at, avatar, face_image_path 
        FROM users 
        WHERE role != 'admin' AND is_active = 1 
        ORDER BY id DESC LIMIT 5
    """).fetchall()
    
    recent_users = []
    for row in recent_users_rows:
        u = dict(row)
        # Ensure created_at is just date
        if u['created_at']:
            u['created_at'] = u['created_at'][:10]
        if not u['face_image_path']:
            u['face_image_path'] = get_user_face_image_path(u['user_id'], u['role'])
        recent_users.append(u)
    
    conn.close()
    
    return jsonify({
        'success': True,
        'stats': {
            'total_students': total_students,
            'total_teachers': total_teachers,
            'total_admins': total_admins,
            'total_members': total_students + total_teachers, # Exclude admin locally too
            'overall_attendance_rate': overall_attendance_rate,
            'low_attendance_students': low_attendance_students,
            'pending_requests': pending_requests,
        },
        'recent_activities': recent_users
    })

@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory('dataset', filename)
if __name__ == '__main__':
    app.run(debug=True)
