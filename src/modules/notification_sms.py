from twilio.rest import Client
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
ACCOUNT_SID = os.getenv("ACCOUNT_SID")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
FROM_NUMBER = os.getenv("FROM_NUMBER")
TO_NUMBER = os.getenv("TO_NUMBER")

def notification_sms(
    body: str,    
    ACCOUNT_SID = ACCOUNT_SID,
    AUTH_TOKEN = AUTH_TOKEN,
    FROM_NUMBER = FROM_NUMBER,
    TO_NUMBER = TO_NUMBER
):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    message = client.messages.create(
        body=body,
        from_=FROM_NUMBER,
        to=TO_NUMBER
    )

    return message.sid

"""
body = "test sms"
notification_sms(body)
"""
