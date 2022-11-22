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

open_eye = glob('./dataset/image/open_eye/*.png')
close_eye = glob('./dataset/image/close_eye/*.jpg')
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

EAR_data = []
is_Closed = []

rt_point_x = []
rt_point_y = []
rb_point_x = []
rb_point_y = []
lt_point_x = []
lt_point_y = []
lb_point_x = []
lb_point_y = []
horizonL_point_x = []
horizonL_point_y = []
horizonR_point_x = []
horizonR_point_y = []

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(open_eye):
    image = cv2.imread(file)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    rt = results.multi_face_landmarks[0].landmark[158]
    rb = results.multi_face_landmarks[0].landmark[153]
    lt = results.multi_face_landmarks[0].landmark[160]
    lb = results.multi_face_landmarks[0].landmark[144]
    hl = results.multi_face_landmarks[0].landmark[130]
    hr = results.multi_face_landmarks[0].landmark[243]

    p1 = point_dist(rt, rb)
    p2 = point_dist(lt, lb)
    p3 = point_dist(hr, hl)

    rt_point_x.append(round(rt.x, 4))
    rt_point_y.append(round(rt.y, 4))
    rb_point_x.append(round(rb.x, 4))
    rb_point_y.append(round(rb.y, 4))
    lt_point_x.append(round(lt.x, 4))
    lt_point_y.append(round(lt.y, 4))
    lb_point_x.append(round(lb.x, 4))
    lb_point_y.append(round(lb.y, 4))
    horizonL_point_x.append(round(hl.x, 4))
    horizonL_point_y.append(round(hl.y, 4))
    horizonR_point_x.append(round(hr.x, 4))
    horizonR_point_y.append(round(hr.y, 4))

    EAR_data.append(round(calculate_EAR(p1,p2,p3),4))
    is_Closed.append(0)

  for idx, file in enumerate(close_eye):
    image = cv2.imread(file)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    rt = results.multi_face_landmarks[0].landmark[158]
    rb = results.multi_face_landmarks[0].landmark[153]
    lt = results.multi_face_landmarks[0].landmark[160]
    lb = results.multi_face_landmarks[0].landmark[144]
    hl = results.multi_face_landmarks[0].landmark[130]
    hr = results.multi_face_landmarks[0].landmark[243]

    p1 = point_dist(rt, rb)
    p2 = point_dist(lt, lb)
    p3 = point_dist(hr, hl)

    rt_point_x.append(round(rt.x, 4))
    rt_point_y.append(round(rt.y, 4))
    rb_point_x.append(round(rb.x, 4))
    rb_point_y.append(round(rb.y, 4))
    lt_point_x.append(round(lt.x, 4))
    lt_point_y.append(round(lt.y, 4))
    lb_point_x.append(round(lb.x, 4))
    lb_point_y.append(round(lb.y, 4))
    horizonL_point_x.append(round(hl.x, 4))
    horizonL_point_y.append(round(hl.y, 4))
    horizonR_point_x.append(round(hr.x, 4))
    horizonR_point_y.append(round(hr.y, 4))

    EAR_data.append(round(calculate_EAR(p1,p2,p3),4))
    is_Closed.append(1)
  
  df = pd.DataFrame(EAR_data, columns=['EAR'])
  df['left_horizon_x'] = horizonL_point_x
  df['left_horizon_y'] = horizonL_point_y
  df['right_horizon_x'] = horizonR_point_x
  df['right_horizon_y'] = horizonR_point_y
  df['left_top_x'] = lt_point_x
  df['left_top_y'] = lt_point_y
  df['left_bottom_x'] = lb_point_x
  df['left_bottom_y'] = lb_point_y
  df['right_top_x'] = rt_point_x
  df['right_top_y'] = rt_point_y
  df['right_bottom_x'] = rb_point_x
  df['right_bottom_y'] = rb_point_y
  df['is_closed'] = is_Closed
  
  df.to_csv("./eye_Ear_Dataset.csv", index = False)