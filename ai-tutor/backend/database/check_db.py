import sqlite3
import os

print(f"CWD: {os.getcwd()}")
db_path = 'attendance.db'
if not os.path.exists(db_path):
    print("attendance.db not found here")
else:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    print("--- PENDING REGISTRATIONS ---")
    rows = c.execute("SELECT id, name, email, registration_status, face_image_path FROM pending_registrations").fetchall()
    for row in rows:
        print(dict(row))
        exists = os.path.exists(os.path.join('..', row['face_image_path']))
        print(f"Image exists on disk: {exists}")
        
    print("\n--- USERS ---")
    users = c.execute("SELECT user_id, name, email, face_image_path FROM users ORDER BY created_at DESC LIMIT 5").fetchall()
    for u in users:
        print(dict(u))
        
    conn.close()
