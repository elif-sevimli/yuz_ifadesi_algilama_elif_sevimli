import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pandas as pd
import os


ifade = input("Hangi ifadeyi topluyorsunuz? (mutlu, uzgun, kizgin, saskin): ")


csv_dosya = 'veriseti.csv'

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kamera açılırken hata.")
        break

    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        row = []

        for lm in landmarks:
            row.append(lm.x)
            row.append(lm.y)

        
        row.append(ifade)

        
        if not os.path.exists(csv_dosya):
            
            kolonlar = []
            for i in range(len(landmarks)):
                kolonlar.append(f"x{i+1}")
                kolonlar.append(f"y{i+1}")
            kolonlar.append("ifade")

            df = pd.DataFrame(columns=kolonlar)
            df.loc[len(df)] = row
            df.to_csv(csv_dosya, index=False)
        else:
            
            df = pd.DataFrame([row])
            df.to_csv(csv_dosya, mode='a', header=False, index=False)

    
    cv2.putText(frame, f'IFADE: {ifade.upper()}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('Veri Toplama', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
