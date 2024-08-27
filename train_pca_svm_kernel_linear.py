
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, recall_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
import random

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
  model = SVC(kernel='linear')
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

# Normalizar os dados
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))


n_components = 50
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

model = train(X_train_pca, y_train)

print('Resultados de Treino PCA x SVM')
predict_and_evaluate(model, X_train_pca, y_train)
print('Resultados de Teste PCA x SVM')
predict_and_evaluate(model, X_test_pca, y_test)
