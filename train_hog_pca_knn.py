
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, recall_score, roc_auc_score
from skimage.feature import hog
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
import random

def extract_hog_features(images):
    hog_features = []
    hog_images = []
    for image in images:
        features, img = hog(image, orientations=9,      # o número de direções angulares distintas para as quais os gradientes são calculados
                              pixels_per_cell=(8, 8),   # tamanho (em pixels) de cada célula na qual a imagem é dividida para calcular o histograma de gradientes orientados
                              cells_per_block=(2, 2),
                              visualize=True)          # índice do eixo do canal na imagem de entrada.
        hog_features.append(features)
        hog_images.append(img)
    return np.array(hog_features), hog_images

def visualize_multiple_hog(images, hog_images, images_to_show=5):
    plt.figure(figsize=(6, images_to_show))

    for i in range(images_to_show):
        # Mostrar imagem original
        plt.subplot(images_to_show, 2, 2 * i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Imagem Original {i+1}', fontsize=8)
        plt.axis('off')
        plt.savefig(str(i + 1) + '_hog.png')

        # Mostrar imagem HOG
        plt.subplot(images_to_show, 2, 2 * i + 2)
        plt.imshow(hog_images[i], cmap='gray')
        plt.title(f'Imagem HOG {i+1}', fontsize=8)
        plt.axis('off')
        plt.savefig(str(i + 2) + '_hog.png')    

def predict_and_evaluate(model, X_test, y_test):
    # Inferência
    y_pred = model.predict(X_test)
    
    # Métricas
    print('Acurácia:', accuracy_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred, average='weighted'))
    print('Recall Score:', recall_score(y_test, y_pred))
    print('ROC AUC Score:', roc_auc_score(y_test, y_pred))
    print('Cohen Kappa Score:', cohen_kappa_score(y_test, y_pred))

    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

def train(X_train, y_train):
  model = KNeighborsClassifier(n_neighbors=15,
    weights='distance',
    algorithm='auto',
    leaf_size=60,
    p=2,
    metric='euclidean',
    n_jobs=-1)
  model.fit(X_train, y_train)
  return model

def load_images_from_folder(folder, img_size, labels_dict=None, max_images=None, sort=False):
    images = []
    labels = []

    file_list = os.listdir(folder)
    # Ordenar os arquivos numericamente (por conta do submission file)
    if sort:
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))

    for filename in file_list:
        if max_images and len(images) >= max_images:
            break
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, 1) #1- Color 0 - grayscale
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(rgb_img, (img_size, img_size))

            img_array = np.array(resized_image)
            images.append(img_array)
            # Etiquetar as imagens: 0 para gato, 1 para cachorro
            if labels_dict:
                labels.append(labels_dict[filename])
            elif 'parasitized' in filename:
                labels.append(0)
            elif 'uninfected' in filename:
                labels.append(1)
    return np.array(images), np.array(labels)

train_folder = 'train'
img_size = 64  # Redimensionar imagens para 64x64 pixels

# Carregar imagens de treino
X_train, y_train = load_images_from_folder(train_folder, img_size)

# Para o conjunto de teste, podemos simplesmente carregar as imagens sem etiquetas
# ou usar um conjunto de validação a partir dos dados de treinamento
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train_hog, train_hog_images = extract_hog_features(X_train)
X_test_hog, _ = extract_hog_features(X_test)

# Exemplo de visualização das primeiras n imagens do conjunto de treinamento
images_to_show = 5
visualize_multiple_hog(X_train, train_hog_images, images_to_show=images_to_show)

plt.tight_layout()

# Normalizar os dados
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
#
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

n_components = 15
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)


# Concatenar as características HOG e PCA
X_train_concat = np.hstack((X_train_hog, X_train_pca))
X_test_concat = np.hstack((X_test_hog, X_test_pca))

model = train(X_train_concat, y_train)

print('Resultados de Treino HOG x PCA x KNN')
predict_and_evaluate(model, X_train_concat, y_train)
print('Resultados de Teste HOG x PCA x KNN')
predict_and_evaluate(model, X_test_concat, y_test)
