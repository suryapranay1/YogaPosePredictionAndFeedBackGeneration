import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3

engine = pyttsx3.init()
speech_played = False  # Global flag to track whether speech has been played

engine = pyttsx3.init()
speech_played = False  # Global flag to track whether speech has been played

def text2speech(text_file):
    global speech_played

    if not speech_played:  # Check if speech has not been played yet
        try:
            with open(text_file, 'r') as f:
                text = f.read()

            engine.say(text)
            engine.runAndWait()
            speech_played = True  # Set flag to indicate speech has been played
        except Exception as e:
            print("Error during text-to-speech conversion:", e)



diffe=[]
from sklearn.preprocessing import StandardScaler
tree_np=np.load('Tree_angles.npy')
with open('body_language000.pkl', 'rb') as f:
    model = pickle.load(f)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

angles = []
mp_pose = mp.solutions.pose.Pose()
mpDraw=mp.solutions.drawing_utils
mp_pose1 = mp.solutions.pose
pose=mp_pose1.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(r"C:\Users\surya\PycharmProjects\yfg\.venv\pxl-20240328-045014113ts_v7YGfrBD.mp4")
n_features = 132
scaler = StandardScaler()
scaler.fit(np.zeros((1, n_features)))
try:
 while True:
    ret, frame = cap.read()
    sucess,img=cap.read()
    #imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5 )
    results=pose.process(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        pose_results = mp_pose.process(image=frame)
        #pose_results = mp_pose.process(image=frame)
        if pose_results.pose_landmarks is not None:
            lmd = pose_results.pose_landmarks.landmark
            lmd = results.pose_landmarks.landmark
            Rshoulder = [lmd[mp_pose1.PoseLandmark.RIGHT_SHOULDER.value].x,lmd[mp_pose1.PoseLandmark.RIGHT_SHOULDER.value].y]
            Relbow = [lmd[mp_pose1.PoseLandmark.RIGHT_ELBOW.value].x, lmd[mp_pose1.PoseLandmark.RIGHT_ELBOW.value].y]
            Rwrist = [lmd[mp_pose1.PoseLandmark.RIGHT_WRIST.value].x, lmd[mp_pose1.PoseLandmark.RIGHT_WRIST.value].y]
            Rangle = calculate_angle(Rshoulder, Relbow, Rwrist)
            Lshoulder = [lmd[mp_pose1.PoseLandmark.LEFT_SHOULDER.value].x, lmd[mp_pose1.PoseLandmark.LEFT_SHOULDER.value].y]
            Lelbow = [lmd[mp_pose1.PoseLandmark.LEFT_ELBOW.value].x, lmd[mp_pose1.PoseLandmark.LEFT_ELBOW.value].y]
            Lwrist = [lmd[mp_pose1.PoseLandmark.LEFT_WRIST.value].x, lmd[mp_pose1.PoseLandmark.LEFT_WRIST.value].y]
            Langle = calculate_angle(Lshoulder, Lelbow, Lwrist)
            RHip = [lmd[mp_pose1.PoseLandmark.RIGHT_HIP.value].x, lmd[mp_pose1.PoseLandmark.RIGHT_HIP.value].y]
            RKnee = [lmd[mp_pose1.PoseLandmark.RIGHT_KNEE.value].x, lmd[mp_pose1.PoseLandmark.RIGHT_KNEE.value].y]
            RAnkle = [lmd[mp_pose1.PoseLandmark.RIGHT_ANKLE.value].x, lmd[mp_pose1.PoseLandmark.RIGHT_ANKLE.value].y]
            Rangle_leg = calculate_angle(RHip, RKnee, RAnkle)
            LHip = [lmd[mp_pose1.PoseLandmark.LEFT_HIP.value].x, lmd[mp_pose1.PoseLandmark.LEFT_HIP.value].y]
            LKnee = [lmd[mp_pose1.PoseLandmark.LEFT_KNEE.value].x, lmd[mp_pose1.PoseLandmark.LEFT_KNEE.value].y]
            LAnkle = [lmd[mp_pose1.PoseLandmark.LEFT_ANKLE.value].x, lmd[mp_pose1.PoseLandmark.LEFT_ANKLE.value].y]
            Langle_leg = calculate_angle(LHip, LKnee, LAnkle)
            angles_list=[Rangle,Langle,Rangle_leg,Langle_leg]
            angles_np=np.array(angles_list)
            difference = angles_np - tree_np
            print(angles_np)
            print(tree_np)
            print(difference)

        mpDraw.draw_landmarks(img, results.pose_landmarks, mp_pose1.POSE_CONNECTIONS)

    finally:
        pass
    if pose_results.pose_landmarks is not None:
        pose_row = list(np.array(
            [[float(landmark.x), float(landmark.y), float(landmark.z), float(landmark.visibility)] for landmark in
             lmd]).flatten())
        row = pose_row
    else:
        row = np.zeros(n_features)
    row=np.array(row)

    row = row.reshape(1, -1)
    prediction = model.predict(row)
    print(prediction)
    for i in prediction:
        pred=i
    if(pred=='tree'):
        label = "tree"
        text2speech("tree.txt")
    elif(pred=="cobra"):
        label="cobra"
        text2speech("cobra.txt")
    elif (pred == "corpse"):
        label = "corpse"
        text2speech("corpse.txt")
    elif (pred == "mountain"):
        label = "mountain"
        text2speech("mountain.txt")
    elif (pred == "triangle"):
        label = "triangle"
        text2speech("triangle.txt")
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    np_load=np.load(pred+'_angles.npy')
    diff=angles_np-np_load
    print(diff)
    diffe=[]
    if abs(diff[0])>30:
       diffe.append('right hand is not aligned correctly')
    if abs(diff[1])>30:
        diffe.append('left hand is not aligned correctly')
    if abs(diff[2])>45:
        diffe.append('right leg is not aligned correctly')
    if abs(diff[3])>30:
        diffe.append('left leg is not aligned correctly')
    print(diffe)
    # Assuming frame is your image frame where you're displaying the text
    y_position = 100
    for text in diffe:
        cv2.putText(frame, text, (50, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_position += 30
    # First line of text
    cv2.putText(frame, str(prediction[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Second line of text


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
except:
    pass

cap.release()
cv2.destroyAllWindows()
