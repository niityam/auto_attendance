import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from utils import (
    detector,
    embedder,
    extract_face,
    get_embedding,
    load_embeddings,
    save_embeddings,
    load_dataset,
    normalize_embeddings,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import load_model, Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import pickle
import tensorflow as tf

# Set TensorFlow log level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Streamlit app title
st.title('Automatic Attendance System')

# Sidebar menu
menu = ['Home', 'Add New Student', 'Train Model', 'Mark Attendance']
choice = st.sidebar.selectbox('Menu', menu)

# Global variables
DATASET_DIR = 'dataset'
EMBEDDINGS_PATH = 'embeddings/embeddings.pkl'
CHECKPOINT_DIR = 'checkpoint'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'classifier_checkpoint.keras')
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
ATTENDANCE_LOG = 'attendance_logs/attendance.csv'

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs('embeddings', exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('attendance_logs', exist_ok=True)

# Function to train the classifier model
def train_classifier_model():
    st.write('Preparing data...')
    # Load or create embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        data = load_embeddings(EMBEDDINGS_PATH)
        embeddings, labels = data['embeddings'], data['labels']
    else:
        X, y = load_dataset(DATASET_DIR)
        embeddings = [get_embedding(face) for face in X]
        embeddings = np.array(embeddings)
        data = {'embeddings': embeddings, 'labels': y}
        save_embeddings(EMBEDDINGS_PATH, data)
        labels = y
    # Normalize embeddings
    embeddings = normalize_embeddings(embeddings)
    # Encode labels
    out_encoder = LabelEncoder()
    out_encoder.fit(labels)
    labels_enc = out_encoder.transform(labels)
    # Save label encoder
    with open(LABEL_ENCODER_PATH, 'wb') as file:
        pickle.dump(out_encoder, file)
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels_enc, test_size=0.2, random_state=42
    )
    num_classes = len(out_encoder.classes_)
    # Create or load the model
    if os.path.exists(CHECKPOINT_PATH):
        model = load_model(CHECKPOINT_PATH)
        st.write('Loaded existing model checkpoint.')
    else:
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(embeddings.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
        )
        st.write('Created new model.')
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1,
    )
    # Train the model
    st.write('Training the model...')
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[checkpoint_callback],
    )
    st.success('Model training completed and best model saved!')

# Home page
if choice == 'Home':
    st.write('Welcome to the Automatic Attendance System. Use the sidebar to navigate.')

# Add New Student
elif choice == 'Add New Student':
    st.header('Add New Student')
    enrollment_number = st.text_input('Enter Enrollment Number')
    uploaded_files = st.file_uploader(
        'Upload Student Images', accept_multiple_files=True, type=['jpg', 'png', 'jpeg']
    )
    if st.button('Add Student'):
        if enrollment_number and uploaded_files:
            student_dir = os.path.join(DATASET_DIR, enrollment_number)
            os.makedirs(student_dir, exist_ok=True)
            for uploaded_file in uploaded_files:
                image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image_path = os.path.join(student_dir, uploaded_file.name)
                cv2.imwrite(image_path, image)
            st.success(f'Student {enrollment_number} added successfully!')
            st.info('Please navigate to "Train Model" to update the classifier.')
        else:
            st.error('Please provide an enrollment number and upload at least one image.')

# Train Model
elif choice == 'Train Model':
    st.header('Train Classifier Model')
    if st.button('Start Training'):
        train_classifier_model()

# Mark Attendance
elif choice == 'Mark Attendance':
    st.header('Mark Attendance')
    uploaded_class_image = st.file_uploader(
        'Upload Classroom Image', type=['jpg', 'png', 'jpeg']
    )
    if st.button('Process Image'):
        if uploaded_class_image:
            # Load label encoder
            if not os.path.exists(LABEL_ENCODER_PATH):
                st.error('Model not trained. Please train the model first.')
                st.stop()
            with open(LABEL_ENCODER_PATH, 'rb') as file:
                out_encoder = pickle.load(file)
            num_classes = len(out_encoder.classes_)
            # Load classifier model
            if not os.path.exists(CHECKPOINT_PATH):
                st.error('Model not trained. Please train the model first.')
                st.stop()
            model = load_model(CHECKPOINT_PATH)
            # Read the uploaded image
            image = np.array(bytearray(uploaded_class_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Detect faces
            faces = detector.detect_faces(image_rgb)
            attendance_list = []
            for face in faces:
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face_img = image_rgb[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (160, 160))
                embedding = get_embedding(face_img)
                embedding = normalize_embeddings([embedding])
                # Predict class
                yhat = model.predict(embedding)
                class_index = np.argmax(yhat, axis=1)[0]
                class_probability = yhat[0, class_index] * 100
                predicted_label = out_encoder.inverse_transform([class_index])[0]
                # Threshold for confidence
                threshold = 70
                if class_probability > threshold:
                    attendance_list.append(predicted_label)
                    # Draw bounding box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        f'{predicted_label} ({class_probability:.2f}%)',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        image,
                        'Unknown',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
            # Save attendance log
            if attendance_list:
                df = pd.DataFrame({
                    'Enrollment Number': attendance_list,
                    'Timestamp': pd.Timestamp.now(),
                })
                if os.path.exists(ATTENDANCE_LOG):
                    df.to_csv(ATTENDANCE_LOG, mode='a', header=False, index=False)
                else:
                    df.to_csv(ATTENDANCE_LOG, index=False)
                st.success('Attendance marked successfully!')
            else:
                st.warning('No recognized faces detected.')
            # Display the processed image
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels='RGB')
            # Display attendance list
            st.subheader('Attendance List')
            for enrollment_number in set(attendance_list):
                st.write(enrollment_number)
        else:
            st.error('Please upload a classroom image.')
