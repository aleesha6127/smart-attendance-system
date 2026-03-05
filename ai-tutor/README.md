# AI-Based Smart Attendance & Learning System

A comprehensive attendance and learning management system using Python, Flask, face recognition, and AI-powered tutoring.

## 🚀 Features

### User Roles
- **Admin**: System management, user management, analytics
- **Teacher**: Attendance marking, class management
- **Student**: Attendance tracking, learning assistance

### Core Features
- Face recognition-based attendance
- Role-based access control
- AI tutor with MCA syllabus knowledge
- Attendance analytics and reporting
- Secure password management

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Database**: SQLite
- **Face Recognition**: OpenCV, face_recognition
- **AI Tutor**: Transformers, Sentence Transformers, RAG
- **Frontend**: HTML, CSS, JavaScript, Bootstrap

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-tutor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup script:
```bash
python setup.py
```

## 🏃‍♂️ Running the Application

1. Start the Flask server:
```bash
cd backend
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

## 👤 Default Credentials

- **Admin**: 
  - User ID: `admin001`
  - Password: `admin123`

## 🏗️ Project Structure

```
backend/
├── app.py                 # Main Flask application
├── face_recognition.py    # Face recognition module
├── ai_tutor.py           # AI tutor with RAG
├── database/
│   └── attendance.db     # SQLite database
├── dataset/
│   ├── students/         # Student face images
│   └── teachers/         # Teacher face images
├── models/
│   └── face_encodings.pkl # Trained face encodings
├── rag/
│   ├── vector_store/     # Vector store for RAG
│   └── mca_syllabus/     # MCA syllabus documents
└── templates/            # HTML templates
    ├── base.html
    ├── login.html
    ├── admin/
    ├── teacher/
    └── student/
```

## 🎯 Usage

### For Admins
1. Log in with admin credentials
2. Add teachers and students via "Add User"
3. Register faces for users
4. Monitor system analytics

### For Teachers
1. Mark attendance using face recognition or manual entry
2. Access AI tutor for teaching assistance

### For Students
1. View attendance records
2. Request attendance corrections
3. Access AI tutor for learning assistance

## 🤖 AI Tutor Capabilities

The AI tutor is powered by Retrieval-Augmented Generation (RAG) and trained specifically on MCA syllabus materials. It can:
- Answer questions about programming concepts
- Explain algorithms and data structures
- Discuss database systems and software engineering
- Provide explanations on computer networks and more

## 🔐 Security Features

- Password hashing using SHA-256
- Session management
- Role-based access control
- No raw biometric storage (only face encodings)

## 📊 Analytics & Reporting

- Attendance percentages
- Student engagement metrics
- Risk analysis for low attendance
- Department-wise summaries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support, please contact the development team or create an issue in the repository.