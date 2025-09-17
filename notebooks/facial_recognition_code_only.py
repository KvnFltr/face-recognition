#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import cv2
import dlib
import os
import keras
import sklearn
import random
import shutil

from keras import layers
from keras import models
from keras import optimizers


# In[ ]:


def suppr_models():
    get_ipython().system('rm models.zip')
    get_ipython().system('rm models -r')

def suppr_figures():
    get_ipython().system('rm figures.zip')
    get_ipython().system('rm figures -r')

def suppr_data():
    get_ipython().system('rm data.zip')
    get_ipython().system('rm data -r')
    get_ipython().system('rm __MACOSX -r')

suppr_models()
suppr_figures()
suppr_data()

get_ipython().system('wget -q https://perso.esiee.fr/~najmanl/FaceRecognition/models.zip')
get_ipython().system('unzip -q models.zip')
get_ipython().system('wget -q https://perso.esiee.fr/~najmanl/FaceRecognition/figures.zip')
get_ipython().system('unzip -q figures.zip')
get_ipython().system('wget -q https://perso.esiee.fr/~najmanl/FaceRecognition/data.zip')
get_ipython().system('unzip -q data.zip')


# In[ ]:


hog_detector = dlib.get_frontal_face_detector()
cnn_detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')

def face_locations(image, model="hog"):

    if model == "hog":
        detector = hog_detector
        cst = 0
    elif model == "cnn":
        detector = cnn_detector
        cst = 10

    matches = detector(image,1)
    rects   = []

    for r in matches:
        if model == "cnn":
            r = r.rect
        x = max(r.left(), 0)
        y = max(r.top(), 0)
        w = min(r.right(), image.shape[1]) - x + cst
        h = min(r.bottom(), image.shape[0]) - y + cst
        rects.append((x,y,w,h))

    return rects

def extract_faces(image, model="hog"):

    gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = face_locations(gray, model)
    faces = []

    for (x,y,w,h) in rects:
        cropped = image[y:y+h, x:x+w, :]
        cropped = cv2.resize(cropped, (128,128))
        faces.append(cropped)

    return faces

def show_grid(faces, figsize=(12,3)):

    n = len(faces)
    cols = 7
    rows = int(np.ceil(n/cols))

    fig, ax = plt.subplots(rows,cols, figsize=figsize)

    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            if i == n:
                 break
            ax[r,c].imshow(faces[i])
            ax[r,c].axis('off')
            #ax[r,c].set_title('size: ' + str(faces[i].shape[:2]))

def list_images(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):

    imagePaths = []

    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                imagePaths.append(imagePath)

    return imagePaths

base_dir = "data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")


# In[ ]:


image = cv2.imread("figures/faces.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15,5))
plt.imshow(image)

faces = extract_faces(image, "cnn")  # Replace 'cnn' with 'hog' for faster but less accurate results
show_grid(faces)


# In[ ]:


# Observation des données

get_ipython().system('ls data/alan_grant/')
# Visualisation d'une image des données pour se faire une idée
impath = "data/alan_grant/00000082.jpg"
if os.path.exists(impath):
    image = cv2.imread(impath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)


# In[ ]:


# 1. Extraction des faces dans les mêmes répertoires data/prenom_nom

# Liste tous les chemins d'images dans le répertoire "data"
imagePaths = list_images("data")

for imagePath in imagePaths:
    # Si le nom de l'image contient le mot "face", on passe à l'image suivante
    if "face" in os.path.basename(imagePath): 
        continue

    # Lecture de l'image à partir du chemin
    image = cv2.imread(imagePath)

    # Extraction des visages de l'image
    # Utilisation du modèle "cnn" pour une meilleure détection de face, bien que "hog" soit plus rapide 
    faces = extract_faces(image, model="cnn")

    # Si aucun visage n'est détecté, on passe à l'image suivante
    if len(faces) == 0:
        continue

    # Comme il n'y a jamais plus d'un visage par image, on manipule directement le premier visage détectée
    face = faces[0]

    # Récupération du chemin complet en remplaçant l'extension (.png, .jpg...) par "_face.jpg"
    face_filename = os.path.splitext(imagePath)[0] + "_face.jpg"

    # Enregistrement du visage dans le même répertoire que l'image d'origine
    cv2.imwrite(face_filename, face)

print("Extraction des visages terminée.")


# 2. Réarrangement des visages dans des répertoires d'entraînement, de validation et de test

# Création des répertoires
for path in [train_dir, val_dir, test_dir]:
    if os.path.exists(path):
        shutil.rmtree(path) # suppression des répertoires déjà existants
    os.makedirs(path, exist_ok=True)

# Récupération des répertoires prenom_nom dans data
person_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
               and d not in ["train", "validation", "test"]]

for person in person_dirs:
    person_path = os.path.join(base_dir, person)

    # Utilisation de list_images pour récupérer les images qui terminent par "face.jpg"
    images = list_images(person_path, contains="_face.jpg")

    # Mélange aléatoire des images
    random.shuffle(images)

    # Calcul des indices de découpage pour répartir les images en trois ensembles
    total = len(images)
    train_count = int(total * 0.65)  # % des images dans les données d'entraînement
    val_count = int(total * 0.2)  # % des images pour la validation

    # Découpage de l'ensemble des images en trois groupes : entraînement, validation et test
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Définition des chemins de destination pour chaque ensemble
    person_train_dir = os.path.join(train_dir, person)
    person_val_dir = os.path.join(val_dir, person)
    person_test_dir = os.path.join(test_dir, person)

    # Création des répertoires de destination s'ils n'existent pas déjà
    for path in [person_train_dir, person_val_dir, person_test_dir]:
        os.makedirs(path, exist_ok=True)

    # Déplacement des images dans les bons répertoires
    for img in train_images:
        shutil.move(img, os.path.join(person_train_dir, os.path.basename(img)))
    for img in val_images:
        shutil.move(img, os.path.join(person_val_dir, os.path.basename(img)))
    for img in test_images:
        shutil.move(img, os.path.join(person_test_dir, os.path.basename(img)))

print("Répartition des visages dans les répertoires train, validation et test terminée.")


# In[ ]:


# Création des générateurs

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=10,
    class_mode='categorical' # categorical puisqu'on a 6 variables catégorielles (les différents personnages)
)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=10,          # Ajusté en f° du nombre d'images dans les données de validation
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=10,
    class_mode='categorical'
)


# In[ ]:


for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)
    print('data label shape:', labels_batch.shape)

    plt.imshow(data_batch[0])
    plt.show()

    break


# In[ ]:


# Création et entraînement du réseau de neurones

from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(6, activation='softmax')) # softmax vu qu'on a 6 sorties

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=1e-4), metrics=['acc'])


# In[ ]:


history = model.fit(
    train_generator,  # Use train_generator for training data
    epochs=30,
    validation_data=validation_generator  # Use validation_generator for validation data
)


# In[ ]:


# Get the training info
loss     = history.history['loss']
val_loss = history.history['val_loss']
acc      = history.history['acc']
val_acc  = history.history['val_acc']

# Visualize the history plots
plt.figure()
plt.plot(loss, 'b', label='Training loss')
plt.plot(val_loss, 'm', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
plt.figure()
plt.plot(acc, 'b', label='Training acc')
plt.plot(val_acc, 'm', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()


# In[ ]:


val_loss, val_acc = model.evaluate(validation_generator, steps=20)
print('Validation accuracy: {:2.2f}%'.format(val_acc*100))
test_loss, test_acc = model.evaluate(test_generator, steps=20)
print('Test accuracy: {:2.2f}%'.format(test_acc*100))


# In[ ]:


pose68 = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
pose05 = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

def face_landmarks(face, model="large"):

    if model == "large":
        predictor = pose68
    elif model == "small":
        predictor = pose05

    if not isinstance(face, list):
        rect = dlib.rectangle(0,0,face.shape[1],face.shape[0])
        return predictor(face, rect)
    else:
        rect = dlib.rectangle(0,0,face[0].shape[1],face[0].shape[0])
        return [predictor(f,rect) for f in face]

def shape_to_coords(shape):
    return np.float32([[p.x, p.y] for p in shape.parts()])

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

INNER_EYES_AND_BOTTOM_LIP = np.array([39, 42, 57])
OUTER_EYES_AND_NOSE = np.array([36, 45, 33])


def align_faces(images, landmarks, idx=INNER_EYES_AND_BOTTOM_LIP):
    faces = []
    for (img, marks) in zip(images, landmarks):
        imgDim = img.shape[0]
        coords = shape_to_coords(marks)
        H = cv2.getAffineTransform(coords[idx], imgDim * MINMAX_TEMPLATE[idx])
        warped = cv2.warpAffine(img, H, (imgDim, imgDim))
        faces.append(warped)
    return faces


# In[ ]:


landmarks = face_landmarks(faces)

new_faces = []
for (face,shape) in zip(faces, landmarks):
    canvas = face.copy()
    coords = shape_to_coords(shape)
    for p in coords:
        cv2.circle(canvas, (int(p[0]),int(p[1])), 1, (0, 0, 255), -1)
    new_faces.append(canvas)

show_grid(new_faces, figsize=(15,5))

aligned = align_faces(faces, landmarks)
show_grid(aligned, figsize=(15,5))


# In[ ]:


plt.imshow( np.stack(aligned, axis=3).astype(np.float32).mean(axis=3)/255 )


# In[ ]:


# Conversion de chaque visages du répertoire data en visages centrés
# Chaque visage est traité individuellement
def center_faces(base_dir):
    # Lister toutes les images terminant par "_face.jpg"
    imagePaths = list_images(base_dir, contains="_face.jpg")

    for imagePath in imagePaths:
        # Lire l'image
        image = cv2.imread(imagePath)

        # Détecter les points de repère
        landmarks = face_landmarks(image)

        # Aligner le visage
        aligned_face = align_faces([image], [landmarks])[0]

        # Construire le chemin pour sauvegarder l'image centrée
        newImagePath = imagePath.replace("_face.jpg", "_centered.jpg")

        # Sauvegarder l'image centrée
        cv2.imwrite(newImagePath, aligned_face)

        # Supprimer l'image originale (_face.jpg)
        os.remove(imagePath)
        
    print("Traitement des images terminé.")

center_faces(base_dir)


# In[ ]:


ls data/train/alan_grant/


# In[ ]:


for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)
    print('data label shape:', labels_batch.shape)

    plt.imshow(data_batch[0])
    plt.show()

    break


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=1e-4), metrics=['acc'])


# In[ ]:


history = model.fit(
    train_generator,  # Use train_generator for training data
    epochs=30,
    validation_data=validation_generator  # Use validation_generator for validation data
)


# In[ ]:


# Get the training info
loss     = history.history['loss']
val_loss = history.history['val_loss']
acc      = history.history['acc']
val_acc  = history.history['val_acc']

# Visualize the history plots
plt.figure()
plt.plot(loss, 'b', label='Training loss')
plt.plot(val_loss, 'm', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
plt.figure()
plt.plot(acc, 'b', label='Training acc')
plt.plot(val_acc, 'm', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()


# In[ ]:


validation_loss, validation_accuracy = model.evaluate(validation_generator)
print('Validation accuracy: {:2.2f}%'.format(validation_accuracy * 100))

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy: {:2.2f}%'.format(test_acc*100))


# In[ ]:


cnn_encoder = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def face_encoder(faces):

    landmarks = face_landmarks(faces)

    if not isinstance(faces, list):
        return np.array(cnn_encoder.compute_face_descriptor(faces,landmarks))
    else:
        return np.array([cnn_encoder.compute_face_descriptor(f,l) for f,l in zip(faces,landmarks)])


encoded_faces = face_encoder(faces)

plt.plot(encoded_faces[0])


# In[ ]:


def encode_faces(directory):
    data = []
    labels = []

    # Lister tous les sous-répertoires dans le répertoire donné
    for person_dir in os.listdir(directory):
        person_path = os.path.join(directory, person_dir)

        # Lister toutes les images terminant par "_centered.jpg"
        image_paths = [os.path.join(person_path, file_name) for file_name in os.listdir(person_path) if file_name.endswith('_centered.jpg')]
        # Lire toutes les images
        images = [cv2.imread(image_path) for image_path in image_paths]

        # Encoder les visages
        encoded_faces = face_encoder(images)

        # Ajouter les vecteurs encodés et les labels aux listes
        data.extend(encoded_faces)
        labels.extend([person_dir] * len(encoded_faces))


    return np.array(data), np.array(labels)


# In[ ]:


from sklearn.preprocessing import LabelBinarizer


train_data, train_labels = encode_faces(train_dir)
val_data, val_labels = encode_faces(val_dir)
test_data, test_labels = encode_faces(test_dir)

# Encodage one-hot des labels
label_binarizer = LabelBinarizer()
train_labels_onehot = label_binarizer.fit_transform(train_labels)
val_labels_onehot = label_binarizer.transform(val_labels)
test_labels_onehot = label_binarizer.transform(test_labels)


# In[ ]:


# création du réseau
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(128,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(train_labels_onehot.shape[1], activation='softmax')) # REMARK: softmax is for multi-class classification

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit(train_data, train_labels_onehot, validation_data=(val_data, val_labels_onehot), epochs=40, batch_size=32)


# In[ ]:


# Get the training info
loss     = history.history['loss']
val_loss = history.history['val_loss']
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']

# Visualize the history plots
plt.figure()
plt.plot(loss, 'b', label='Training loss')
plt.plot(val_loss, 'm', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
plt.figure()
plt.plot(acc, 'b', label='Training acc')
plt.plot(val_acc, 'm', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()


# In[ ]:


# Évaluer le modèle sur l'ensemble de validation
val_loss, val_accuracy = model.evaluate(val_data, val_labels_onehot)
print('Validation accuracy: {:2.2f}%'.format(val_accuracy * 100))

# Évaluer le modèle sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(test_data, test_labels_onehot)
print('Test accuracy: {:2.2f}%'.format(test_accuracy * 100))


# In[ ]:


def process_frame(image, mode="fast", model=None):

    # face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "fast":
        matches = hog_detector(gray,1)
    else:
        matches = cnn_detector(gray,1)
        matches = [m.rect for m in matches]

    for rect in matches:

        # face classification
        if model is None:
            label = "label"
        else:
            # face landmarks
            landmarks = pose68(gray, rect)
             # face encoding
            encoding = cnn_encoder.compute_face_descriptor(image, landmarks)
            # Convert the encoding to the correct shape for prediction
            encoding = np.array(encoding).reshape(1, -1)
            # Predict the label
            prediction = model.predict(encoding)
            label = prediction[0]
        
        # draw box
        cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
        y = rect.top() - 15 if rect.top() - 15 > 15 else rect.bottom() + 25
        cv2.putText(image, label, (rect.left(), y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return image

def process_movie(video_name, outvideo_name='/kaggle/working/videoResult.mp4', mode="fast", model=None):

    video  = cv2.VideoCapture(video_name)
    if (video.isOpened()== False): 
        print("Error opening video stream or file")
        return
    
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    out_mp4 = cv2.VideoWriter(outvideo_name,cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width//2,frame_height//2))

    i=0
    while video.isOpened():

        # Grab a single frame of video
        ret, frame = video.read()
        if ret == True:
            # Resize frame of video for faster processing
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # Process frame
            image = process_frame(frame, mode, model)
            # Write the processed frame to output video
            out_mp4.write(image)
        else:
            break
        i += 1
        if i==1000:
            break
    # Release video
    video.release()
    out_mp4.release()
    print("Video released")


from IPython.display import HTML
from base64 import b64encode

def play(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=1000 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)


def suppr_test():
    get_ipython().system('rm test.zip')
    get_ipython().system('rm test -r')
    get_ipython().system('rm __MACOSX -r')

suppr_test()

get_ipython().system('wget -q https://perso.esiee.fr/~najmanl/FaceRecognition/test.zip')
get_ipython().system('unzip -q test.zip')
get_ipython().system('ls test')


# In[ ]:


# Logistic regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(train_data, train_labels)

# SVM classifier
from sklearn.svm import SVC

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(train_data, train_labels)

# kNN classifier
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(train_data, train_labels)


# In[ ]:


# ÉVALUATION DES MODÈLES
import time
from sklearn.metrics import accuracy_score

def evaluate_model(model, train_data, train_labels, test_data, test_labels):
    start_time = time.time()
    predictions = model.predict(test_data)
    prediction_time = time.time() - start_time
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy, prediction_time

def evaluate_keras_model(model, train_data, train_labels, test_data, test_labels):
    start_time = time.time()
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    prediction_time = time.time() - start_time
    return accuracy, prediction_time

logistic_accuracy, logistic_pred_time = evaluate_model(logistic_model, train_data, train_labels, test_data, test_labels)
print(f'Logistic Regression - Accuracy: {logistic_accuracy:.2f}, Prediction Time: {logistic_pred_time:.4f}s')

svm_accuracy, svm_pred_time = evaluate_model(svm_model, train_data, train_labels, test_data, test_labels)
print(f'SVM - Accuracy: {svm_accuracy:.2f}, Prediction Time: {svm_pred_time:.4f}s')

knn_accuracy, knn_pred_time = evaluate_model(knn_model, train_data, train_labels, test_data, test_labels)
print(f'kNN - Accuracy: {knn_accuracy:.2f}, Prediction Time: {knn_pred_time:.4f}s')

nn_accuracy, nn_pred_time = evaluate_keras_model(model, train_data, train_labels_onehot, test_data, test_labels_onehot)
print(f'Neural Network - Accuracy: {nn_accuracy:.2f}, Prediction Time: {nn_pred_time:.4f}s')
# model : réseau entraîné dans la partie précédente


# In[ ]:


# Test sur une des images fournie
image = cv2.imread("test/example_03.png")
processed = process_frame(image.copy(), model=logistic_model)
processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15,5))
plt.imshow(processed)


# In[ ]:


# Test sur la vidéo fournie
get_ipython().system('rm videoResult.mp4')
get_ipython().system('rm videoResult_fixed.mp4')
process_movie("test/lunch_scene.mp4", mode="not fast", model=logistic_model)
# Reconvertir la vidéo avec ffmpeg en .mp4 compatible
get_ipython().system('ffmpeg -i videoResult.mp4 -vcodec libx264 -acodec aac videoResult_fixed.mp4 >/dev/null 2>&1')
play('/kaggle/working/videoResult_fixed.mp4')


# In[ ]:


# Cette partie peut être exécutée indépendamment des parties pécédentes


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import cv2
import dlib
import os
import keras
import sklearn
import random
import shutil

from keras import layers
from keras import models
from keras import optimizers


# In[ ]:


def suppr_models():
    get_ipython().system('rm models.zip')
    get_ipython().system('rm models -r')

def suppr_figures():
    get_ipython().system('rm figures.zip')
    get_ipython().system('rm figures -r')

def suppr_data():
    get_ipython().system('rm data.zip')
    get_ipython().system('rm data -r')
    get_ipython().system('rm __MACOSX -r')

suppr_models()
suppr_figures()
suppr_data()

# importation des modèles préentraînés fournis
get_ipython().system('wget -q https://perso.esiee.fr/~najmanl/FaceRecognition/models.zip')
get_ipython().system('unzip -q models.zip')


hog_detector = dlib.get_frontal_face_detector()
cnn_detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat') # nécessite un GPU

# fonctions utilitaires
def face_locations(image, model="hog"):

    if model == "hog":
        detector = hog_detector
        cst = 0
    elif model == "cnn":
        detector = cnn_detector
        cst = 10

    matches = detector(image,1)
    rects   = []

    for r in matches:
        if model == "cnn":
            r = r.rect
        x = max(r.left(), 0)
        y = max(r.top(), 0)
        w = min(r.right(), image.shape[1]) - x + cst
        h = min(r.bottom(), image.shape[0]) - y + cst
        rects.append((x,y,w,h))

    return rects

def extract_faces(image, model="hog"):

    gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = face_locations(gray, model)
    faces = []

    for (x,y,w,h) in rects:
        cropped = image[y:y+h, x:x+w, :]
        cropped = cv2.resize(cropped, (128,128))
        faces.append(cropped)

    return faces

def show_grid(faces, figsize=(12,3)):

    n = len(faces)
    cols = 7
    rows = int(np.ceil(n/cols))

    fig, ax = plt.subplots(rows,cols, figsize=figsize)

    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            if i == n:
                 break
            ax[r,c].imshow(faces[i])
            ax[r,c].axis('off')
            #ax[r,c].set_title('size: ' + str(faces[i].shape[:2]))

def list_images(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):

    imagePaths = []

    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                imagePaths.append(imagePath)

    return imagePaths

pose68 = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
pose05 = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

def face_landmarks(face, model="large"):

    if model == "large":
        predictor = pose68
    elif model == "small":
        predictor = pose05

    if not isinstance(face, list):
        rect = dlib.rectangle(0,0,face.shape[1],face.shape[0])
        return predictor(face, rect)
    else:
        rect = dlib.rectangle(0,0,face[0].shape[1],face[0].shape[0])
        return [predictor(f,rect) for f in face]

def shape_to_coords(shape):
    return np.float32([[p.x, p.y] for p in shape.parts()])

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

INNER_EYES_AND_BOTTOM_LIP = np.array([39, 42, 57])
OUTER_EYES_AND_NOSE = np.array([36, 45, 33])


def align_faces(images, landmarks, idx=INNER_EYES_AND_BOTTOM_LIP):
    faces = []
    for (img, marks) in zip(images, landmarks):
        imgDim = img.shape[0]
        coords = shape_to_coords(marks)
        H = cv2.getAffineTransform(coords[idx], imgDim * MINMAX_TEMPLATE[idx])
        warped = cv2.warpAffine(img, H, (imgDim, imgDim))
        faces.append(warped)
    return faces


base_dir = "data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")


# In[ ]:


# Au choix :

# importation du jeu de données brut
get_ipython().system('pip install -q gdown')
import gdown

file_id = "1I6GZo2uU3d6r51pzbfmJlE7LhQ-R2GKr"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, output="data.zip", quiet=False)

get_ipython().system('unzip -q data.zip')

# Extraction des visages dans les mêmes répertoires data/prenom_nom

imagePaths = list_images("data")
for imagePath in imagePaths:
    if "face" in os.path.basename(imagePath): 
        continue
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Erreur : Impossible de lire l'image {imagePath}. Elle sera ignorée.")
        continue
    faces = extract_faces(image, model="cnn")
    if len(faces) == 0:
        print(f"Erreur : Aucun visage détecté dans l'image {imagePath}. Elle sera ignorée.")
        continue
    if len(faces) > 1:
        print(f"Erreur : Plus d'un visage détecté dans l'image {imagePath}. Elle sera ignorée.")
        continue
    face = faces[0]
    face_filename = os.path.splitext(imagePath)[0] + "_face.jpg"
    cv2.imwrite(face_filename, face)
print("Extraction des visages terminée.")


# In[ ]:


# Ou bien :
# Charger les visages déjà extraits (pour gagner du temps)
suppr_data()

get_ipython().system('pip install -q gdown')
import gdown
file_id = "1OXezr6FgQIitVeKSzHaqtr4WUIQfOwIM"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, output="data_faces.zip", quiet=False)
get_ipython().system('unzip -q data_faces.zip')


# In[ ]:


# Répartition des visages dans des répertoires d'entraînement, de validation et de test

# Création des répertoires
for path in [train_dir, val_dir, test_dir]:
    if os.path.exists(path):
        shutil.rmtree(path) # suppression des répertoires déjà existants
    os.makedirs(path, exist_ok=True)
# Récupération des répertoires prenom_nom dans data
person_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
               and d not in ["train", "validation", "test"]]

for person in person_dirs:
    person_path = os.path.join(base_dir, person)

    # Utilisation de list_images pour récupérer les images qui terminent par "face.jpg"
    images = list_images(person_path, contains="_face.jpg")

    # Mélange aléatoire des images
    random.shuffle(images)

    # Calcul des indices de découpage pour répartir les images en trois ensembles
    total = len(images)
    train_count = int(total * 0.7)  # % des images dans les données d'entraînement
    val_count = int(total * 0.15)  # % des images pour la validation

    # Découpage de l'ensemble des images en trois groupes : entraînement, validation et test
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Définition des chemins de destination pour chaque ensemble
    person_train_dir = os.path.join(train_dir, person)
    person_val_dir = os.path.join(val_dir, person)
    person_test_dir = os.path.join(test_dir, person)

    # Création des répertoires de destination s'ils n'existent pas déjà
    for path in [person_train_dir, person_val_dir, person_test_dir]:
        os.makedirs(path, exist_ok=True)

    # Déplacement des images dans les bons répertoires
    for img in train_images:
        shutil.move(img, os.path.join(person_train_dir, os.path.basename(img)))
    for img in val_images:
        shutil.move(img, os.path.join(person_val_dir, os.path.basename(img)))
    for img in test_images:
        shutil.move(img, os.path.join(person_test_dir, os.path.basename(img)))

print("Répartition des visages dans les répertoires train, validation et test terminée.")


# In[ ]:




