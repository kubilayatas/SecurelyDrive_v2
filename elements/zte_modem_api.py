import requests
import base64
import urllib.parse
from datetime import datetime, timezone
    
def turkish_to_unicode(message):
    if message.find("ğ") != -1 : message = message.replace('ğ', '\u011f')
    if message.find("Ğ") != -1 : message = message.replace('Ğ', '\u011e')
    if message.find("ı") != -1 : message = message.replace('ı', '\u0131')
    if message.find("İ") != -1 : message = message.replace('İ', '\u0130')
    if message.find("ö") != -1 : message = message.replace('ö', '\u00f6')
    if message.find("Ö") != -1 : message = message.replace('Ö', '\u00d6')
    if message.find("ü") != -1 : message = message.replace('ü', '\u00fc')
    if message.find("Ü") != -1 : message = message.replace('Ü', '\u00dc')
    if message.find("ş") != -1 : message = message.replace('ş', '\u015f')
    if message.find("Ş") != -1 : message = message.replace('Ş', '\u015e')
    if message.find("ç") != -1 : message = message.replace('ç', '\u00e7')
    if message.find("Ç") != -1 : message = message.replace('Ç', '\u00c7')
    return message
    
class zte_modem():
    
    def __init__(self,modem_ip,passwd):
        self.modem_ip = modem_ip
        self.header = {"Referer": "http://{}/index.html".format(modem_ip)}
        self.url = 'http://{}/goform/goform_set_cmd_process'.format(modem_ip)
        self.passwd64 = base64.b64encode(passwd.encode("ascii")).decode()
        self.login_data = {"isTest":"false","goformId":"LOGIN","password":self.passwd64}
        login = requests.post(self.url, data = self.login_data, headers = self.header)
        print("ZTE modem login: "+login.text)
    
    
    def send_sms(self,number,message):
        now = datetime.now(timezone.utc)
        date = now.strftime("%y;%m;%d;%H;%M;%S;") + "+3"
        date = urllib.parse.quote_plus(date)
        message = turkish_to_unicode(message)
        message = message.encode("utf-8").hex().upper()
        hex_message = []
        for i in range(len(message)):
            if i%2 == 0:
                hex_message.append("00")
                hex_message.append(message[i])
            else:
                hex_message.append(message[i])
        message = "".join(hex_message)
        send_sms_dat = {"isTest" : "false", "goformId" : "SEND_SMS", "notCallback" : "true", "Number" : "{}".format(number), "sms_time" : "{}".format(date), "MessageBody" : "{}".format(message), "ID" : "-1", "encode_type" : "GSM7_turkey"}
        send_sms = requests.post(self.url, data = send_sms_dat, headers = self.header)
        print("SMS sending: "+send_sms.text)
    