import cv2
import numpy as np
import time
import csv
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__, template_folder='.')  # Set template_folder to the root directory

# Initialize MediaPipe Pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables to store report data
counter = 0
correct_counter = 0
incorrect_counter = 0
reason = ""

# Helper function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Check for body alignment
def check_body_alignment(shoulder, hip, knee):
    angle = calculate_angle(shoulder, hip, knee)
    return angle > 160

# Check for swinging motion
def check_swinging_motion(previous_time, current_time):
    return (current_time - previous_time) < 0.5

# Check for incomplete range of motion
def check_incomplete_range_of_motion(angle):
    return not (angle < 30 or angle > 160)

def gen():
    global counter, correct_counter, incorrect_counter, reason
    cap = cv2.VideoCapture(0)
    stage = None
    rep_started = False
    start_time = None
    rep_counted = False
    previous_rep_time = time.time()

    start_time = time.time()
    duration = 15

    # Reset counters for a new session
    counter = 0
    correct_counter = 0
    incorrect_counter = 0
    reason = ""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with open('exercise_data.csv', mode='w', newline='') as exercise_data_file:
            csv_writer = csv.writer(exercise_data_file)
            csv_writer.writerow(['Timestamp', 'Rep Count', 'Stage', 'Elbow Close', 'Straight Wrist', 'Body Alignment', 'Swinging Motion', 'Incomplete Range of Motion'])

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    angle = calculate_angle(shoulder, elbow, wrist)

                    elbow_close = angle < 30
                    straight_wrist = wrist[1] > shoulder[1]
                    current_time = time.time()
                    swinging_motion = check_swinging_motion(previous_rep_time, current_time)
                    incomplete_range_of_motion = check_incomplete_range_of_motion(angle)
                    body_alignment = check_body_alignment(shoulder, hip, knee)

                    incorrect_message = []
                    if swinging_motion:
                        incorrect_message.append("Swinging motion: Avoid abrupt movements.")
                    if incomplete_range_of_motion:
                        incorrect_message.append("Incomplete range of motion: Ensure full extension and contraction.")
                    if not body_alignment:
                        incorrect_message.append("Body alignment: Keep shoulder, hip, and knee aligned.")
                    if not elbow_close:
                        incorrect_message.append("Ensure your elbows are close to your torso.")
                    if not straight_wrist:
                        incorrect_message.append("Ensure your shoulders are unshrugged.")

                    if angle > 160:
                        stage = "down"
                        if not rep_started:
                            start_time = time.time()
                            rep_started = True
                            rep_counted = False
                    if angle < 30 and stage == 'down':
                        stage = "up"
                        if rep_started:
                            rep_duration = current_time - start_time
                            time_between_reps = current_time - previous_rep_time
                            if rep_duration > 0.5 and not rep_counted:
                                if not swinging_motion and not incomplete_range_of_motion and body_alignment and elbow_close and straight_wrist:
                                    counter += 1
                                    correct_counter += 1
                                    csv_writer.writerow([current_time, counter, stage, elbow_close, straight_wrist, body_alignment, not swinging_motion, not incomplete_range_of_motion])
                                else:
                                    incorrect_counter += 1
                                    for msg in incorrect_message:
                                        reason += f"Incorrect Rep {incorrect_counter}: {msg}\n"
                                rep_counted = True
                                previous_rep_time = current_time
                            elif time_between_reps < 1.5:
                                incorrect_counter += 1
                                reason += f"Incorrect Rep {incorrect_counter}: You're performing the exercise too quickly.\n"

                        rep_started = False

                    if incorrect_message:
                        for idx, msg in enumerate(incorrect_message):
                            cv2.putText(image, msg, (10, 30 + (idx * 30)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                           
                except:
                    pass

                # Improve the font settings for displayed text
                cv2.putText(image, 'REPS', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (15, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)  # Green color
                
                cv2.putText(image, 'STAGE', (15, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, stage if stage else 'None', 
                            (15, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)  # Blue color
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               

                ret, jpeg = cv2.imencode('.jpg', image)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

                elapsed_time = time.time() - start_time
                if elapsed_time > duration:
                    break

            cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/report')
def report():
    global counter, correct_counter, incorrect_counter, reason
    return jsonify({
        'Total Reps': counter + incorrect_counter,
        'Correct Reps': correct_counter,
        'Incorrect Reps': incorrect_counter,
        'Reason for Incorrect Reps': reason
    })

if __name__ == '__main__':
    app.run(debug=True)
