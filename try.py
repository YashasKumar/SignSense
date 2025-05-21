import mediapipe as mp
import cv2
import numpy as np
from keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
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

def predict_value(model, frames, keypoints, actions, threshold=0.01):
    # Add a batch dimension to frames and keypoints
    frames = np.expand_dims(frames, axis=0)  # Shape: (1, 16, 128, 128, 3)
    keypoints = np.expand_dims(keypoints, axis=0)  # Shape: (1, 16, num_keypoints)

    # Predict with the model
    res = model.predict([frames, keypoints])
    max_prob = np.max(res)

    if max_prob > threshold:
        return actions[np.argmax(res[0])]
    else:
        return None

def start():
    cap = cv2.VideoCapture(0)
    frames = []
    keypoints_list = []
    frame_count = 0

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            frame, keypoints = mediapipe_detection(frame, holistic)
            keypoints = process_keys(keypoints)
            frame_count += 1

            # Store frames and keypoints
            frames.append(cv2.resize(frame, (224, 224)))
            keypoints_list.append(keypoints)

            if frame_count == 30:
                model = load_model('finetuned_model3.h5')

                # Adjust frames and keypoints to 16
                frames, keypoints_list = adjust_to_16(frames, keypoints_list)
                frames = np.array(frames)  # Shape: (16, 128, 128, 3)
                keypoints_list = np.array(keypoints_list)  # Shape: (16, num_keypoints, 3)

                # Optionally, compute pairwise distances
                pairwise_distances = np.array([
                    compute_pairwise_distances(keypoints) for keypoints in keypoints_list
                ])  # Shape: (16, num_keypoints, num_keypoints)

                # Ensure the input matches model's expected format
                sign_prediction = predict_value(
                    model,
                    frames,
                    pairwise_distances,  # Use pairwise distances or keypoints_list based on model expectation
                    ['apple', 'basketball', 'bed']
                )

                if sign_prediction:
                    print(sign_prediction)
                    cv2.waitKey(3000)

                # Reset for the next batch
                frames = []
                keypoints_list = []
                frame_count = 0

            # Show the frame
            cv2.imshow('Sign Language Detection', frame)

            # Break loop on 'q' key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start()