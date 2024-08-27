
import os

path = 'C:\\trabalho_visao_computacional\\temp'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, str(index).join(['parasitized.', '.png'])))

print('arquivos renomeados com sucesso.')