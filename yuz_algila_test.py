import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import joblib
import numpy as np


model = joblib.load("model.pkl")


base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


emoji_dict = {
    "mutlu": "😊",
    "uzgun": "😢",
    "kizgin": "😠",
    "saskin": "😲"
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kamera açılamadı.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    detection_result = detector.detect(mp_image)

    ifade_tahmini = "Bilinmiyor"

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        data = []

        for lm in landmarks:
            data.append(lm.x)
            data.append(lm.y)

        data_np = np.array(data).reshape(1, -1)
        tahmin = model.predict(data_np)[0]

        ifade_tahmini = tahmin

    
    emoji = emoji_dict.get(ifade_tahmini, "")
    metin = f"IFADE: {ifade_tahmini.upper()} {emoji}"
    cv2.putText(frame, metin, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Gerçek Zamanlı Yüz İfadesi Tanıma", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
