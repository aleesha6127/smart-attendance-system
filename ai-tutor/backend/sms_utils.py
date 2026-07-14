import os
import requests
from datetime import datetime
from dotenv import load_dotenv

def send_credentials_sms(phone_number, user_id, password, role="user"):
    """
    Sends an SMS using Twilio.
    """
    load_dotenv(override=True)
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    twilio_number = os.environ.get('TWILIO_PHONE_NUMBER')

    formatted_number = str(phone_number).strip()
    if not formatted_number.startswith('+'):
        if len(formatted_number) == 10:
            formatted_number = "+91" + formatted_number # Default to India
        else:
             formatted_number = "+" + formatted_number

    message_body = f"AI Tutor: {role.title()} approved.\nID: {user_id}\nPass: {password}"
    
    # --- Try Twilio FIRST ---
    if all([account_sid, auth_token, twilio_number]):
        try:
            from twilio.rest import Client
            from twilio.base.exceptions import TwilioRestException
            client = Client(account_sid, auth_token)
            message = client.messages.create(
                body=message_body,
                from_=twilio_number,
                to=formatted_number
            )
            print(f"[SMS] Sent via Twilio to {formatted_number}. SID: {message.sid}")
            return True, "SMS sent successfully via Twilio."
        except TwilioRestException as e:
            print(f"[SMS Error] Twilio Failed: {e.msg if hasattr(e, 'msg') else str(e)}")
            # Fall through to next method
        except Exception as e:
            print(f"[SMS Error] Unknown Error: {str(e)}")
            # Fall through to next method
    else:
        print("[SMS Warning] Missing Twilio credentials or old credentials in cache. Falling back.")

    # --- Fallback 1. Try Textbelt (Free Zero-Config API, 1 per day) ---
    try:
        resp = requests.post('https://textbelt.com/text', {
            'phone': formatted_number,
            'message': message_body,
            'key': 'textbelt',
        }, timeout=5)
        data = resp.json()
        if data.get('success'):
            print(f"[SMS] Sent via Free Textbelt API to {formatted_number}")
            return True, "SMS sent successfully (Free API)."
        else:
            reason = data.get('error', 'Quota exceeded')
            print(f"[SMS] Textbelt failed: {reason}")
    except Exception as e:
        print(f"[SMS Error] Textbelt API request failed: {e}")

    # --- Fallback 2. Local File Simulation ---
    try:
        os.makedirs("dataset", exist_ok=True)
        log_file = "dataset/sms_simulation_log.txt"
        with open(log_file, "a") as f:
            f.write(f"\n--- SMS SIMULATION [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
            f.write(f"TO: {formatted_number}\n")
            f.write(f"MESSAGE:\n{message_body}\n")
            f.write("-" * 40 + "\n")
        
        print(f"[SMS] SIMULATED delivery to {formatted_number}. Written to {log_file}")
        return True, "SMS successfully simulated (Check dataset/sms_simulation_log.txt)"
    except Exception as e:
        return False, f"Failed to simulate SMS: {e}"
