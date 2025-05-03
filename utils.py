import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=1,
                                   min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_face_landmarks(image, draw=False, static_image_mode=True):
    image = cv2.resize(image, (192, 192))
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_input_rgb)
    image_landmarks = []

    if results.multi_face_landmarks:
        if draw:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        landmarks = results.multi_face_landmarks[0].landmark
        xs_ = [pt.x for pt in landmarks]
        ys_ = [pt.y for pt in landmarks]
        zs_ = [pt.z for pt in landmarks]

        min_x, min_y, min_z = min(xs_), min(ys_), min(zs_)
        for x, y, z in zip(xs_, ys_, zs_):
            image_landmarks.extend([x - min_x, y - min_y, z - min_z])

    return image_landmarks
