import requests

url = 'https://edxtfidf.s3.amazonaws.com/EDX.csv'

respoonse = requests.get(url)
if respoonse.status_code == 200:
    with open('EdX.csv','wb') as file:
        file.write(respoonse.content)
    print('Archivo descargado con exito')
else:
    print('Error al descargar el archivo:',respoonse.status_code)