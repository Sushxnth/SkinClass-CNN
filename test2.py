import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Configuration
IMG_SIZE = (256, 256)
MODEL_PATH = 'best_model_cpu.h5'
TEST_IMAGE = r"c:\IEEE HACKATHON\Generated_images_zip\dermatofibroma\dermatofibroma_0021.png"
CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Squamous cell carcinoma',
    'Vascular Lesion'
]
def predict_image(model, image_path, class_names):
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        # Predict
        preds = model.predict(img_array)[0]
        predicted_class_idx = np.argmax(preds)
        predicted_class = class_names[predicted_class_idx]
        
        # Print probabilities
        print(f"Image: {os.path.basename(image_path)}")
        for class_name, prob in zip(class_names, preds):
            print(f"{class_name}: {prob*100:.2f}%")
        print(f"✅ Predicted Class: {predicted_class}")
        
        return predicted_class, preds
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None, None

def main():
    try:
        # Load model from HDF5 file
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        
        # Test image prediction
        if os.path.exists(TEST_IMAGE):
            predicted_class, _ = predict_image(model, TEST_IMAGE, CLASS_NAMES)
            print(f"Predicted Disease: {predicted_class}")
        else:
            print(f"Test image {TEST_IMAGE} not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()