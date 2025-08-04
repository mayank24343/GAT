import cv2 
import numpy as np 
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_landmark(image):
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)
    
    image.flags.writeable = False
     
    result = face_mesh.process(image)
    landmark = result.multi_face_landmarks[0].landmark

    return result,landmark

def draw_landmark(image, result):
    image.flags.writeable = True
    if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=draw_specs,
                    connection_drawing_spec=draw_specs)
    


while True:
    _, frame = cap.read()

    
    a,b = get_landmark(frame)
    draw_landmark(frame,a)

    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1)
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()