from twilio.rest import Client

def notification_sms(
    account_sid: str,
    auth_token: str,
    from_number: str,
    to_number: str,
    body: str
):
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=body,
        from_=from_number,
        to=to_number
    )

    return message.sid