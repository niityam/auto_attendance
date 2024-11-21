import os
import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
from sklearn.preprocessing import Normalizer
import pickle
from keras_facenet import FaceNet

# Initialize the MTCNN face detector
detector = MTCNN()

# Load the FaceNet model using keras-facenet
embedder = FaceNet()

def extract_face(image, required_size=(160, 160)):
    # Detect faces
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    # Extract bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # Correct potential negative values
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # Extract the face
    face = image[y1:y2, x1:x2]
    # Resize to the required size
    face = cv2.resize(face, required_size)
    return face

def load_faces(directory):
    faces = []
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        # Read the image
        image = cv2.imread(path)
        if image is None:
            continue
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = extract_face(image)
        if face is not None:
            faces.append(face)
    return faces

def load_dataset(directory):
    X, y = [], []
    # Iterate over each class directory
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
    return np.array(X), np.array(y)

def get_embedding(face_pixels):
    # Scale pixel values
    face_pixels = face_pixels.astype('float32')
    # Standardize pixel values across channels
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # Get embedding
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = embedder.embeddings(samples)
    return yhat[0]

def save_embeddings(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_embeddings(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def normalize_embeddings(embeddings):
    in_encoder = Normalizer(norm='l2')
    return in_encoder.transform(embeddings)
