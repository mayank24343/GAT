import cv2 
import numpy as np 
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

def midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

def get_eye_position(frame):
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)
    
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh_points = np.array([
            [int(p.x * w), int(p.y * h)] for p in results.multi_face_landmarks[0].landmark
        ])

        left_eye = mesh_points[LEFT_EYE[0]], mesh_points[LEFT_EYE[1]]
        right_eye = mesh_points[RIGHT_EYE[0]], mesh_points[RIGHT_EYE[1]]
        left_center = midpoint(mesh_points[LEFT_IRIS[0]], mesh_points[LEFT_IRIS[2]])
        right_center = midpoint(mesh_points[RIGHT_IRIS[0]], mesh_points[RIGHT_IRIS[2]])

        lx, ly = left_center
        l_left, l_right = left_eye[0][0], left_eye[1][0]
        gaze_ratio_left = (lx - l_left) / (l_right - l_left + 1e-6)

        if gaze_ratio_left > 2.65:
            return "right"
        elif gaze_ratio_left < 2.55:
            return "left"
        else:
            return "center"
    return "center"

def draw_eye_debug(frame):
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh_points = np.array([
            [int(p.x * w), int(p.y * h)] for p in results.multi_face_landmarks[0].landmark
        ])

        for idx in LEFT_IRIS + RIGHT_IRIS + LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, tuple(mesh_points[idx]), 2, (0, 255, 255), -1)

        left_center = midpoint(mesh_points[LEFT_IRIS[0]], mesh_points[LEFT_IRIS[2]])
        right_center = midpoint(mesh_points[RIGHT_IRIS[0]], mesh_points[RIGHT_IRIS[2]])
        cv2.circle(frame, left_center, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_center, 3, (0, 255, 0), -1)
    return frame

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
    

l = []
while True:
    _, frame = cap.read()

    
    a,b = get_landmark(frame)
    pos = get_eye_position(frame)
    l.append(pos)
    #draw_eye_debug(frame)
    draw_landmark(frame,a)

    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1)
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()
print(l)