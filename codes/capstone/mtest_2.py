import cv2
import mediapipe as mp
import pandas as pd
from scipy.spatial import distance
from glob import glob

def point_dist(p1,p2):
    d = distance.euclidean([p1.x, p1.y], [p2.x, p2.y])
    return d

def calculate_EAR(p1,p2,p3):
	ear = (p1+p2)/(2.0*p3)
	return ear

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

angle = glob('./dataset/headpose/angled/*.jpg')
stand = glob('./dataset/headpose/stand/*.jpg')
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

ratio_data = []
is_angled = []

nose_top_x = []
nose_top_y = []
nose_bottom_x = []
nose_bottom_y = []
face_bottom_x = []
face_bottom_y = []

hdp = 200
nosetp = 6
nosebp = 94 

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(angle):
    image = cv2.imread(file)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    fb = results.multi_face_landmarks[0].landmark[hdp]
    nt = results.multi_face_landmarks[0].landmark[nosetp]
    nb = results.multi_face_landmarks[0].landmark[nosebp]

    p1 = point_dist(nt, nb)
    p2 = point_dist(nb, fb)

    ratio = p1 / p2

    ratio_data.append(round(ratio,4))

    nose_top_x.append(round(nt.x,4))
    nose_top_y.append(round(nt.y,4))
    nose_bottom_x.append(round(nb.x,4))
    nose_bottom_y.append(round(nb.y,4))
    face_bottom_x.append(round(fb.x,4))
    face_bottom_y.append(round(fb.y,4))

    is_angled.append(1)

  for idx, file in enumerate(stand):
    image = cv2.imread(file)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    fb = results.multi_face_landmarks[0].landmark[hdp]
    nt = results.multi_face_landmarks[0].landmark[nosetp]
    nb = results.multi_face_landmarks[0].landmark[nosebp]

    p1 = point_dist(nt, nb)
    p2 = point_dist(nb, fb)

    ratio = p1 / p2

    ratio_data.append(round(ratio,4))

    nose_top_x.append(round(nt.x,4))
    nose_top_y.append(round(nt.y,4))
    nose_bottom_x.append(round(nb.x,4))
    nose_bottom_y.append(round(nb.y,4))
    face_bottom_x.append(round(fb.x,4))
    face_bottom_y.append(round(fb.y,4))

    is_angled.append(0)
  
  df = pd.DataFrame(ratio_data, columns=['ratio'])

  df['nose_top_x'] = nose_top_x
  df['nose_top_y'] = nose_top_y
  df['nose_bottom_x'] = nose_bottom_x
  df['nose_bottom_y'] = nose_bottom_y
  df['face_bottom_x'] = face_bottom_x
  df['face_bottom_y'] = face_bottom_y
  df['is_angled'] = is_angled
  
  df.to_csv("./head_Pose_Dataset.csv", index = False)