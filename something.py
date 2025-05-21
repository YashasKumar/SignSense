import mediapipe as mp
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import pickle
from tensorflow.keras import mixed_precision

mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    if image.dtype != np.uint8:
        image = (image*255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    keypoints = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, keypoints

def process_keys(keypoints):
    # Extract landmarks into numpy arrays
    lh = np.zeros((21, 3))
    rh = np.zeros((21, 3))
    pose_keys = np.zeros((8, 3))

    pose_lands = [
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
        mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
        mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
        mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
    ]

    # Check if left hand landmarks are present
    if keypoints.left_hand_landmarks is not None:
        for i, lm in enumerate(keypoints.left_hand_landmarks.landmark):
            lh[i] = [lm.x, lm.y, lm.z]

    # Check if right hand landmarks are present
    if keypoints.right_hand_landmarks is not None:
        for i, lm in enumerate(keypoints.right_hand_landmarks.landmark):
            rh[i] = [lm.x, lm.y, lm.z]

    # Check if pose landmarks are present
    if keypoints.pose_landmarks is not None:
        for i, land_idx in enumerate(pose_lands):
            landmark = keypoints.pose_landmarks.landmark[land_idx]
            pose_keys[i] = [landmark.x, landmark.y, landmark.z]

    # Combine all keypoints
    all_keypoints = np.concatenate((lh, rh, pose_keys), axis=0)

    # Normalize keypoints
    mask = all_keypoints != 0  # Avoid zero values affecting normalization
    if np.any(mask):
        normalized_keypoints = (all_keypoints - np.min(all_keypoints[mask], axis=0)) / \
                            (np.max(all_keypoints[mask], axis=0) - np.min(all_keypoints[mask], axis=0))
    else:
        normalized_keypoints = all_keypoints
    return normalized_keypoints

def adjust_to_16(frames, keypoints, target_count=16):
    current_count = len(frames)
    if current_count == target_count:
        return frames, keypoints
    adjusted_frames, adjusted_keypoints = [], []
    indices = np.linspace(0, current_count - 1, target_count, dtype=int)
    adjusted_frames = [frames[i] for i in indices]
    adjusted_keypoints = [keypoints[i] for i in indices]
    return adjusted_frames, adjusted_keypoints

def compute_pairwise_distances(keypoints):
    """
    Compute pairwise distances between all keypoints.
    
    Args:
        keypoints: numpy array of shape (num_keypoints, 2) with (x, y) coordinates.
    
    Returns:
        pairwise_distances: numpy array of shape (num_keypoints, num_keypoints).
    """
    diff = keypoints[:, np.newaxis, :] - keypoints[np.newaxis, :, :]  # Shape: (50, 50, 3)
    
    # Compute Euclidean distances
    pairwise_distances = np.linalg.norm(diff, axis=-1)  # Shape: (50, 50)
    
    return pairwise_distances

def adjust_brightness(frame, factor=1.5):
    """
    Adjust the brightness of a frame.
    
    Parameters:
    - frame: np.array, the input image frame.
    - factor: float, the factor to adjust brightness. >1 increases brightness, <1 decreases it.
    
    Returns:
    - Adjusted frame with brightness modified.
    """
    # Convert frame to float32 to prevent overflow during multiplication
    frame = frame.astype(np.float32) * factor
    
    # Clip values to stay in valid range [0, 255] for uint8
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame

def data_augmentation(video_path, start_frame, end_frame, video_name):
    
    original_frames, original_keys = [], []
    flipped_frames, flipped_keys = [], []
    rotated_by_degree_frames, rotated_by_degree_keys = [], []
    rotated_frames, rotated_keys = [], []
    bright_frames, bright_keys = [], []
    blurred_frames, blurred_keys = [], []
    noisy_frames, noisy_keys = [], []
    saturated_frames, saturated_keys = [], []
    contrast_frames, contrast_keys = [], []
    scaled_frames, scaled_keys = [], []

    with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set starting frame
        while cap.isOpened():
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_pos > end_frame:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (128, 128))
            if frame.ndim != 3:
                continue
            
            # Original frames
            original_frames.append(frame)
            _, keypoints = mediapipe_detection(frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    original_keys.append(compute_pairwise_distances(keys))

            # Augmentation: Flipping
            flipped_frame = cv2.flip(frame, 1)
            flipped_frames.append(flipped_frame)
            _, keypoints = mediapipe_detection(flipped_frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    flipped_keys.append(compute_pairwise_distances(keys))

            # Augmentation: Rotation
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            rotated_frames.append(rotated_frame)
            _, keypoints = mediapipe_detection(rotated_frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    rotated_keys.append(compute_pairwise_distances(keys))

            # Augmentation: Rotation by arbitrary degrees
            (h, w) = frame.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 10, 1.0)  # 10 degrees
            rotated_degree_frame = cv2.warpAffine(frame, rotation_matrix, (w, h))
            rotated_by_degree_frames.append(rotated_degree_frame)
            _, keypoints = mediapipe_detection(rotated_degree_frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    rotated_by_degree_keys.append(compute_pairwise_distances(keys))

            # Augmentation: Brightness Adjustment
            bright_frame = adjust_brightness(frame, factor=np.random.uniform(1.2, 1.5))
            bright_frames.append(bright_frame)
            _, keypoints = mediapipe_detection(bright_frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    bright_keys.append(compute_pairwise_distances(keys))

            # Augmentation: Gaussian Blur
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            blurred_frames.append(blurred_frame)
            _, keypoints = mediapipe_detection(blurred_frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    blurred_keys.append(compute_pairwise_distances(keys))

            # Augmentation: Gaussian Noise
            noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
            noisy_frame = cv2.add(frame, noise)
            noisy_frames.append(noisy_frame)
            _, keypoints = mediapipe_detection(noisy_frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    noisy_keys.append(compute_pairwise_distances(keys))

            # Augmentation: Saturation Adjustment
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_frame[..., 1] = np.clip(hsv_frame[..., 1] * np.random.uniform(1.2, 1.5), 0, 255)
            saturated_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
            saturated_frames.append(saturated_frame)
            _, keypoints = mediapipe_detection(saturated_frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    saturated_keys.append(compute_pairwise_distances(keys))

            # Augmentation: Contrast Adjustment
            contrast_frame = cv2.convertScaleAbs(frame, alpha=np.random.uniform(1.2, 1.5), beta=0)
            contrast_frames.append(contrast_frame)
            _, keypoints = mediapipe_detection(contrast_frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    contrast_keys.append(compute_pairwise_distances(keys))

            # Augmentation: Scaling (Zoom-In)
            scale = np.random.uniform(1.1, 1.3)
            scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            cropped_frame = scaled_frame[:128, :128]  # Crop to original size
            scaled_frames.append(cropped_frame)
            _, keypoints = mediapipe_detection(cropped_frame, holistic)
            if keypoints:
                keys = process_keys(keypoints)
                if keys is not None:
                    scaled_keys.append(compute_pairwise_distances(keys))

        # Adjust all augmented frames and keypoints to match the required sequence length
        original_frames, original_keys = adjust_to_16(original_frames, original_keys)
        flipped_frames, flipped_keys = adjust_to_16(flipped_frames, flipped_keys)
        rotated_by_degree_frames, rotated_by_degree_keys = adjust_to_16(rotated_by_degree_frames, rotated_by_degree_keys)
        rotated_frames, rotated_keys = adjust_to_16(rotated_frames, rotated_keys)
        bright_frames, bright_keys = adjust_to_16(bright_frames, bright_keys)
        blurred_frames, blurred_keys = adjust_to_16(blurred_frames, blurred_keys)
        noisy_frames, noisy_keys = adjust_to_16(noisy_frames, noisy_keys)
        saturated_frames, saturated_keys = adjust_to_16(saturated_frames, saturated_keys)
        contrast_frames, contrast_keys = adjust_to_16(contrast_frames, contrast_keys)
        scaled_frames, scaled_keys = adjust_to_16(scaled_frames, scaled_keys)

        cap.release()

    # Create labels for all 10 augmented videos
    labels = [video_name] * 10

    return [
        original_frames, flipped_frames, rotated_frames,
        rotated_by_degree_frames, bright_frames, blurred_frames,
        noisy_frames, saturated_frames, contrast_frames, scaled_frames
    ], [
        original_keys, flipped_keys, rotated_keys,
        rotated_by_degree_keys, bright_keys, blurred_keys,
        noisy_keys, saturated_keys, contrast_keys, scaled_keys
    ], labels

def access_and_process_videos():
    dataset = []
    label_mapping = {}
    # data_needed = ['apple', 'basketball', 'bed', 'candy', 'change', 'cold', 'corn', 'cousin', 'deaf', 'drink']
    data_needed = ['apple', 'basketball', 'bed', 'candy', 'change', 'cold']
    # data_needed = ['apple', 'basketball', 'bed']
    main_dir = os.getcwd()
    labels_path = os.path.join(main_dir, "final_dataset_annots.json")
    annots = pd.read_json(labels_path)

    folder_names = annots['clean_text'].unique()
    data_dir = os.path.join(main_dir, 'dataset')

    for folder_name in folder_names:
            if folder_name in data_needed:
                folder_path = os.path.join(data_dir, folder_name)

                if not os.path.exists(folder_path):
                    print(f"Folder not found: {folder_path}")
                    continue

                label_mapping.setdefault(folder_name, len(label_mapping))
                folder_annots = annots[annots['clean_text'] == folder_name]

                j = 0
                
                for (_, row) in folder_annots.iterrows():
                    if j <23:
                        video_name = str(row['label'])
                        video_path = os.path.join(folder_path, f"{video_name}.mp4")

                        if not os.path.exists(video_path):
                            video_path = os.path.join(folder_path, f"{int(video_name):05}.mp4")
                        if not os.path.exists(video_path):
                            video_path = os.path.join(folder_path, f"{video_name}.mov")
                        if not os.path.exists(video_path):
                            video_path = os.path.join(folder_path, f"{int(video_name):05}.mov")

                        if not os.path.exists(video_path):
                            print(f"Video not found: {video_name}")
                            continue

                        start_frame = int(row['start'])
                        end_frame = int(row['end'])
                        frames, keypoints, labs = data_augmentation(video_path, start_frame, end_frame, folder_name)
                            
                        for i in range(len(frames)):
                            dataset.append((np.array(frames[i]), np.array(keypoints[i]), labs[i]))
                        j+=1
    return dataset

# from resnet3d import Resnet3DBuilder

# def build_resnet3d_model(input_shape=(16, 224, 224, 3), num_classes=10):
#     model = Resnet3DBuilder.build_resnet_18(input_shape=input_shape, num_outputs=num_classes)
#     return model

# Set mixed precision policy
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

def build_video_classification_model(num_classes=6):
    """
    Build a video classification model combining 3D ConvNet for spatial features
    and LSTM for temporal modeling with attention mechanism.
    """
    # Input for the video sequence
    video_input = layers.Input(shape=(16, 128, 128, 3), name="video_input")
    
    # 3D Convolutional layers for spatial feature extraction
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(video_input)
    x = layers.MaxPooling3D((1, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Flatten spatial dimensions while keeping the temporal dimension
    x = layers.TimeDistributed(layers.Flatten())(x)  # Shape: (batch, time, features)
    
    # LSTM layers for temporal modeling
    x = layers.LSTM(64, return_sequences=True, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(256, activation='relu')(x)
    
    # Input for keypoints
    keypoints_input = layers.Input(shape=(16, 50, 50), name='keypoints_input')
    keypoints_reshaped = layers.Reshape((16, 50, 50, 1))(keypoints_input)  # Add channel dimension

    # 3D ConvNet for keypoints processing
    y = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(keypoints_reshaped)
    y = layers.MaxPooling3D((1, 2, 2))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.4)(y)
    
    y = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(keypoints_reshaped)
    y = layers.MaxPooling3D((1, 2, 2))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.4)(y)
    
    # LSTM for keypoints temporal modeling
    y = layers.TimeDistributed(layers.Flatten())(y)
    y = layers.LSTM(64, return_sequences=True)(y)
    y = layers.Dropout(0.4)(y)
    y = layers.LSTM(256, activation='relu')(y)
    
    # Combine features
    combined_features = layers.Concatenate()([x, y])  # Combine features from both streams
    
    # Classification Layer
    outputs = layers.Dense(num_classes, activation="softmax")(combined_features)
    
    # Define the final model
    model = Model(inputs=[video_input, keypoints_input], outputs=outputs)
    return model

def create_tf_dataset(frames, keypoints, labels, batch_size=1):
    """
    Create a tf.data.Dataset for loading data efficiently in batches.
    """
    def preprocess_data(frame, keypoint, label):
        return ({"video_input": frame, "keypoints_input": keypoint}, label)

    dataset = tf.data.Dataset.from_tensor_slices((frames, keypoints, labels))
    dataset = dataset.map(preprocess_data).shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def train_model(model, dataset, num_epochs=20, batch_size=1):
    """
    Train the model using TensorFlow's Dataset API.
    """
    # Unpack dataset
    frames, keypoints, labels = zip(*dataset)
    frames = np.array(frames, dtype=np.float32)  # Ensure float32 for compatibility with GPU
    keypoints = np.array(keypoints, dtype=np.float32)  # Ensure float32 for compatibility with GPU
    labels = np.array(labels)  # Ensure float32 for compatibility with GPU

    # Split into train, temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        list(zip(frames, keypoints)), labels, test_size=0.3, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Separate frames and keypoints for datasets
    X_train_frames, X_train_keypoints = zip(*X_train)
    X_val_frames, X_val_keypoints = zip(*X_val)
    X_test_frames, X_test_keypoints = zip(*X_test)

    # Convert labels to one-hot encoding
    label_encoder = LabelEncoder()
    y_train = tf.keras.utils.to_categorical(label_encoder.fit_transform(y_train))
    y_val = tf.keras.utils.to_categorical(label_encoder.transform(y_val))
    y_test = tf.keras.utils.to_categorical(label_encoder.transform(y_test))

    # Create tf.data.Dataset
    train_dataset = create_tf_dataset(np.array(X_train_frames), np.array(X_train_keypoints), y_train, batch_size)
    val_dataset = create_tf_dataset(np.array(X_val_frames), np.array(X_val_keypoints), y_val, batch_size)
    test_dataset = create_tf_dataset(np.array(X_test_frames), np.array(X_test_keypoints), y_test, batch_size)

    # Specify GPU usage (if available)
    with tf.device('/CPU:0'):  # Ensure model operations happen on GPU
        # Load the data into the GPU
        train_dataset = train_dataset.apply(
            tf.data.experimental.prefetch_to_device('/CPU:0')
        )
        val_dataset = val_dataset.apply(
            tf.data.experimental.prefetch_to_device('/CPU:0')
        )
        test_dataset = test_dataset.apply(
            tf.data.experimental.prefetch_to_device('/CPU:0')
        )

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        )
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )

        # Train the model with mixed precision
        # with tf.keras.mixed_precision.experimental.scope():
        history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=num_epochs,
                callbacks=[early_stopping, lr_reducer],
                verbose=2
            )

        # Evaluate on test data
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset, verbose=2)
        print(f"Test Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}")

    return history, label_encoder
    
if __name__ == "__main__":
    
    dataset = access_and_process_videos()
    # print("Dataset loaded")
    
    # with open("dataset6.pkl", "wb") as f:
    #     pickle.dump(dataset, f)
    # print("Saved")

    # Create a model that combines pretrained I3D with keypoints
    model_with_keypoints = build_video_classification_model()
    print("Model initialized")

    with open("dataset6.pkl", "rb") as f:
        dataset = pickle.load(f)
    print("Loaded")

    # Train the model
    train_model(model_with_keypoints, dataset)
    print("Model trained")

    # Save the fine-tuned model
    model_with_keypoints.save("finetuned_model6.h5")
    print("Fine-tuned model saved")