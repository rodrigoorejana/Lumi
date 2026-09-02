import datetime

class SystemInfo:
    def __init__(self):
        pass

    @staticmethod
    def get_time():
        now = datetime.datetime.now()
        # Use :02d for padded two-digit minutes/hours (e.g., 09:05)
        answer = f"Agora são {now.hour} horas e {now.minute:02d} minutos."
        return answer