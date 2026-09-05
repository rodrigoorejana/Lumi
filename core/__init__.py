import os
import datetime
import subprocess

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

    @staticmethod
    def open_explorer() -> str:
        """Opens Windows File Explorer."""
        subprocess.Popen("explorer")
        return "Abrindo o explorador de arquivos."
    
    @staticmethod
    def open_notepad() -> str:
        """Opens the default notepad application."""
        subprocess.Popen("notepad")
        return "Abrindo o bloco de notas."
    @staticmethod
    def close_notepad() -> str:
        """Encerra todas as instâncias do Bloco de Notas no Windows."""
        subprocess.Popen("taskkill /f /im notepad.exe", shell=True)
        return "Fechando o bloco de notas."
    def close_explorer() -> str:
        """Fecha as janelas do Explorador de Arquivos sem fechar a barra de tarefas."""
        cmd = 'powershell -command "(New-Object -ComObject Shell.Application).Windows() | ForEach-Object { $_.Quit() }"'
        subprocess.Popen(cmd, shell=True)
        return "Fechando as janelas do explorador de arquivos."

if __name__ == "__main__":
    print(SystemInfo.get_time())
    print(SystemInfo.get_date())
    print(SystemInfo.open_explorer())
    print(SystemInfo.open_notepad())