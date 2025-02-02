import json


class DataFile:
    filePath = ""
    data=[]


    def __init__(self, filePath):
        self.filePath = filePath
        try:
            self.data = self.read()
        except FileNotFoundError:
            self.write()


    def append(self, hardware, model, parameters, elapsedTime, prompt, answer):
        self.data.append({
            "hardware": hardware,
            "model": model,
            "params": parameters,
            "elapsed time": elapsedTime,
            "prompt": prompt,
            "answer": answer
        })


    def write(self):
        with open(self.filePath, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)


    def read(self):
        with open(self.filePath, 'r', encoding='utf-8') as file:
            return json.load(file)


    def clear(self):
        self.data = []
        self.write()
