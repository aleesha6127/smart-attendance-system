"""
Setup script for AI-Based Smart Attendance & Learning System
This script initializes the database and creates necessary directories
"""

import os
import sys
from backend.app import init_db

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'backend/database',
        'backend/dataset/students',
        'backend/dataset/teachers',
        'backend/models',
        'backend/rag/vector_store',
        'backend/rag/mca_syllabus'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def initialize_database():
    """Initialize the SQLite database with tables and default admin user"""
    print("Initializing database...")
    init_db()
    print("✓ Database initialized successfully")

def install_dependencies():
    """Print instructions for installing dependencies"""
    print("\nTo install dependencies, run:")
    print("pip install -r requirements.txt")

def main():
    print("AI-Based Smart Attendance & Learning System - Setup")
    print("=" * 55)
    
    print("\nCreating directories...")
    create_directories()
    
    print("\nSetting up database...")
    initialize_database()
    
    install_dependencies()
    
    print("\nSetup completed successfully!")
    print("\nTo start the application, run:")
    print("cd backend")
    print("python app.py")
    print("\nThen open your browser and go to http://localhost:5000")
    print("\nDefault admin credentials:")
    print("User ID: admin001")
    print("Password: admin123")

if __name__ == "__main__":
    main()