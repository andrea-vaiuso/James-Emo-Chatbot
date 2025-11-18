import requests

SOUNDS = {
    "enabled": True,       # set False to mute
    "send": "ok",          # "ok", "asterisk", "exclamation", "question", "hand"
    "recv": "asterisk"
}

def play_system_sound(which: str):
    if not SOUNDS["enabled"]:
        return
    try:
        import winsound
        mapping = {
            "ok": winsound.MB_OK,
            "asterisk": winsound.MB_ICONASTERISK,
            "exclamation": winsound.MB_ICONEXCLAMATION,
            "question": winsound.MB_ICONQUESTION,
            "hand": winsound.MB_ICONHAND,
        }
        winsound.MessageBeep(mapping.get(which, winsound.MB_OK))
    except Exception:
        # Non-Windows or any failure: ignore silently
        pass

# A quick location utility
def get_current_location():
    try:
        response = requests.get('https://ipinfo.io', timeout=2.0)
        data = response.json()
        return data['city'] + ", " + data['region'] + ", " + data['country']
    except Exception:
        return "Unknown Location"