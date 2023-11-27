import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('model_file_30epochs.h5')

video = cv2.VideoCapture(0)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Face Detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

positive_emotions = [3, 4, 6]  # Happy, Neutral, Surprise
negative_emotions = [0, 1, 2, 5]  # Angry, Disgust, Fear, Sad

emotion_counts = {'Positive': 0, 'Negative': 0}

# For real-time plotting
fig, ax = plt.subplots()
x_data = []
y_positive = []
y_negative = []

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert BGR image to RGB before processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            x, y, w, h = bbox
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            if label in positive_emotions:
                emotion_text = 'Positive'
            else:
                emotion_text = 'Negative'

            emotion_counts[emotion_text] += 1

    total_faces = sum(emotion_counts.values())
    positive_percentage = (emotion_counts['Positive'] / total_faces) * 100
    negative_percentage = (emotion_counts['Negative'] / total_faces) * 100

    # Update real-time plotting data
    x_data.append(total_faces)
    y_positive.append(positive_percentage)
    y_negative.append(negative_percentage)

    # Plot the data
    ax.clear()
    ax.plot(x_data, y_positive, label='Positive', color='green')
    ax.plot(x_data, y_negative, label='Negative', color='red')
    ax.legend()
    ax.set_xlabel('Total Faces')
    ax.set_ylabel('Percentage')
    ax.set_title('Real-time Emotion Analysis')

    plt.pause(0.1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
plt.show()
