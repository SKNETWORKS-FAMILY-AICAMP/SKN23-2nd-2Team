from twilio.rest import Client
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
ACCOUNT_SID = os.getenv("ACCOUNT_SID")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
FROM_NUMBER = os.getenv("FROM_NUMBER")
TO_NUMBER = os.getenv("TO_NUMBER")

def notification_sms(
    ACCOUNT_SID: str,
    AUTH_TOKEN: str,
    FROM_NUMBER: str,
    TO_NUMBER: str,
    body: str
):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    message = client.messages.create(
        body=body,
        from_=FROM_NUMBER,
        to=TO_NUMBER
    )

    return message.sid

"""
body = "테스트용 메시지입니다."
notification_sms(ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER, TO_NUMBER, body)
"""