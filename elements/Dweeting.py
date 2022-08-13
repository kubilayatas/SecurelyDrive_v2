import requests

class dweeting():
    def __init__(self,thing_name):
        self.url_base = "https://dweet.io/dweet/for/"
        self.thing_name = thing_name
    
    def req_dweet(self,variable,value):
        url = self.url_base
        url += self.thing_name
        url += "?"
        url += variable
        if value != "":
            url += "="
            url += value
        try:
            requests.get(url = url)
        except:
            print("Hata: Dweet gönderilemedi! İnternet bağlantısını kontrol ediniz.")
