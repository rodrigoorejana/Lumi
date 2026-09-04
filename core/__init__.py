import datetime

class SystemInfo:
    def __init__(self):
        pass

    @staticmethod
    def get_time() -> str:
        """
        Returns current time formatted for natural speech synthesis.
        """
        now = datetime.datetime.now()
        hour_str = "uma hora" if now.hour in (1, 13) else f"{now.hour} horas"
        
        if now.minute == 0:
            return f"Agora são {hour_str} em ponto."
        elif now.minute == 1:
            return f"Agora são {hour_str} e um minuto."
        else:
            return f"Agora são {hour_str} e {now.minute} minutos."

    @staticmethod
    def get_date() -> str:
        """
        Returns current date in full Brazilian Portuguese for clear voice output.
        """
        now = datetime.datetime.now()
        
        months = [
            "janeiro", "fevereiro", "março", "abril", "maio", "junho",
            "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
        ]
        
        weekdays = [
            "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
            "sexta-feira", "sábado", "domingo"
        ]
        
        day_name = weekdays[now.weekday()]
        month_name = months[now.month - 1]
        
        return f"Hoje é {day_name}, dia {now.day} de {month_name} de {now.year}."


if __name__ == "__main__":
    print(SystemInfo.get_time())
    print(SystemInfo.get_date())