import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def test_this_model(model):
    with open("dataset6.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Unpack data
    frames, keypoints, labels = zip(*data)
    combined = list(zip(frames, keypoints))
    
    # Split data
    _, X_test, _, y_test = train_test_split(
        combined, labels, test_size=0.2, stratify=labels
    )
    X_test_frame, X_test_keypoints = zip(*X_test)
    X_test_frame = np.array(X_test_frame)
    X_test_keypoints = np.array(X_test_keypoints)

    print(f"Frames Shape: {X_test_frame.shape}, Keypoints Shape: {X_test_keypoints.shape}")

    # Evaluate the model
    label_encoder = LabelEncoder()
    combined_input = [X_test_frame, X_test_keypoints]
    y_test = tf.keras.utils.to_categorical(label_encoder.fit_transform(y_test))
    model.evaluate(combined_input, y_test, verbose = 2)

    # Predict for a single sample
    vals = model.predict([
        np.expand_dims(X_test_frame[0], axis=0),  # Expand batch dimension
        np.expand_dims(X_test_keypoints[0], axis=0)  # Expand batch dimension
    ])
    print(f"Predicted Values: {vals}")
    print(f"True Label: {y_test[0]}")

model = load_model("finetuned_model6.h5")
test_this_model(model)