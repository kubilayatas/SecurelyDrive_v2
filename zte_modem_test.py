import threading
from elements.zte_modem_api import zte_modem

zte_modem = zte_modem("192.168.0.1","admin")

def zte_sms(number,message):
    zte_modem.send_sms("{}".format(number), "{}".format(message))

def send_sms(number,message):
    threading.Thread(target = zte_sms, args=(number,message), daemon = True).start()


send_sms("05356771065","Deneme mesaji")
print("Çalışmaya devam ediyor...")