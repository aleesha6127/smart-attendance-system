# AI Tutor & Attendance System

## System Overview
The **AI Tutor & Attendance System** is a comprehensive solution for managing academic attendance, behavior monitoring, and user administration. It leverages **Face Recognition** (LBPH/Face_Recognition) for automated verification and real-time class monitoring.

---

## Data Flow Diagrams (DFD)

### DFD Level 0: Context Diagram
This diagram represents the entire system as a single process interacting with external entities (Actors).

```mermaid
%%{init: { 'look': 'handDrawn', 'theme': 'neutral' } }%%
graph TD
    %% Entities
    Admin[👤 Admin]
    Teacher[👨‍🏫 Teacher]
    Student[🎓 Student]
    Camera[📷 Video Feed]
    
    %% System
    System(⚙️ AI Tutor & Attendance System)

    %% Admin Interactions
    Admin -- "1. Login Credentials" --> System
    Admin -- "2. User Details (Add/Edit/Delete)" --> System
    Admin -- "3. Face Registration Data" --> System
    System -- "4. Admin Dashboard & Reports" --> Admin

    %% Teacher Interactions
    Teacher -- "5. Login Credentials" --> System
    Teacher -- "6. Mark/Edit Attendance" --> System
    Teacher -- "7. Response to Attendance Requests" --> System
    System -- "8. Teacher Dashboard & Class Stats" --> Teacher
    System -- "9. Real-time Monitoring Alerts" --> Teacher

    %% Student Interactions
    Student -- "10. Login Credentials" --> System
    Student -- "11. Attendance Correction Request" --> System
    System -- "12. Student Dashboard & History" --> Student

    %% Hardware Interactions
    Camera -- "13. Live Video Stream" --> System
```

### DFD Level 1: Process Decomposition
This diagram breaks down the main system into its core sub-processes and data stores.

```mermaid
%%{init: { 'look': 'handDrawn', 'theme': 'neutral' } }%%
graph TD
    %% Entities
    Admin[👤 Admin]
    Teacher[👨‍🏫 Teacher]
    Student[🎓 Student]
    Camera[📷 Camera]

    %% Processes
    P1((1.0 Authentication))
    P2((2.0 User & Face Mgmt))
    P3((3.0 Attendance Mgmt))
    P4((4.0 AI Class Monitor))
    P5((5.0 Reporting & Dashboards))

    %% Data Stores
    D1[(🗄️ Users DB)]
    D2[(🗄️ Attendance DB)]
    D3[(📂 Face Dataset/Encodings)]
    D4[(🗄️ Monitoring/Incidents DB)]

    %% Flow: Authentication
    Admin -->|Creds| P1
    Teacher -->|Creds| P1
    Student -->|Creds| P1
    P1 <-->|Verify Creds| D1
    P1 -->|Session Token| Admin
    P1 -->|Session Token| Teacher
    P1 -->|Session Token| Student

    %% Flow: Admin (User Mgmt)
    Admin -->|User Info/Images| P2
    P2 -->|Save User Profile| D1
    P2 -->|Save Face Encodings| D3

    %% Flow: Attendance
    Teacher -->|Mark Status| P3
    Student -->|Request Correction| P3
    P3 -->|Update Records| D2
    P3 -->|Log Audit Trail| D2
    Teacher -->|Approve Request| P3

    %% Flow: AI Monitoring
    Camera -->|Video Frames| P4
    D3 -->|Load Encodings| P4
    P4 -->|Match Face| D1
    P4 -->|Log Behavior/Session| D4
    P4 -->|Real-time Feedback| Teacher

    %% Flow: Reporting
    D1 --> P5
    D2 --> P5
    D4 --> P5
    P5 -->|View Stats| Admin
    P5 -->|View Daily Summary| Teacher
    P5 -->|View History| Student
```
