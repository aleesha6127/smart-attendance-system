import sqlite3
import os
import json

db_path = 'd:/PROJECTS/MAIN_PROJECT/ai-tutor_new/ai-tutor/backend/database/attendance.db'

with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    rows = [dict(row) for row in c.execute("SELECT id, name, email, registration_status, face_image_path FROM pending_registrations").fetchall()]
    
    users = [dict(row) for row in c.execute("SELECT user_id, name, email, face_image_path FROM users ORDER BY created_at DESC LIMIT 5").fetchall()]

with open('d:/PROJECTS/MAIN_PROJECT/ai-tutor_new/ai-tutor/backend/database/db_dump.json', 'w') as f:
    json.dump({'pending': rows, 'users': users}, f, indent=4)
