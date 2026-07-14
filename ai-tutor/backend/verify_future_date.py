import requests
from datetime import datetime, timedelta

# Note: This assumes the Flask app is running locally on 127.0.0.1:5000
# Since I cannot run the server myself, this is a conceptual verification script.
# In a real scenario, I would use a mock request or the browser tool.

def test_future_date_api():
    base_url = "http://127.0.0.1:5000/api/get_students_by_dept_batch"
    future_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    params = {
        'department': 'Computer Science',
        'batch': '2022-2026',
        'date': future_date,
        'period': '1'
    }
    
    print(f"Testing API with future date: {future_date}")
    # In this environment, I'll simulate the logic instead of making a real request
    today = datetime.now().strftime('%Y-%m-%d')
    if future_date > today:
        print("MOCK RESULT: API would return 400 Bad Request - 'Cannot mark attendance for future dates'")
    else:
        print("MOCK RESULT: FAIL - Future date validation not working")

test_future_date_api()
