from flask import Flask, render_template, Response, jsonify
import threading
import re
import google.generativeai as genai
import cv2
import numpy as np
import mediapipe as mp
from pose_detector import PoseDetector
import openai
import json
from flask import request, redirect, url_for    


app = Flask(__name__, static_url_path='/static')


# Initialize PushupTracker
class PushupTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.camera = cv2.VideoCapture(0)
        self.detector = PoseDetector()
        self.count = 0
        self.feedback = "Fixed Form"
        self.form = 0
        self.error = 0
        self.direction = 0

    def gen_frames(self):
        while True:
            success, frame = self.camera.read()
            if not success:
                break
            else:
                if frame is not None:
                    frame, results = self.detector.find_pose(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = self.pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.pose_landmarks:
                        lm_list = self.detector.find_position(frame, [], results)
                        elbow = self.detector.find_angle(frame, lm_list, 11, 13, 15, results, draw=False)
                        shoulder = self.detector.find_angle(frame, lm_list, 13, 11, 23, results, draw=False)
                        hip = self.detector.find_angle(frame, lm_list, 11, 23, 25, results, draw=False)

                        per = int(np.interp(elbow, (90, 160), (0, 100)))
                        bar = int(np.interp(elbow, (90, 160), (380, 50)))

                        if elbow > 160 and shoulder > 40 and hip > 160:
                            self.form = 1

                        if self.form == 1:
                            if per == 0:
                                if elbow <= 90 and hip > 160:
                                    self.feedback = "Down"
                                    if self.direction == 0:
                                        self.count += 0.5
                                        self.direction = 1
                                else:
                                    self.feedback = "Fix Form"
                                    self.error += 1

                            if per == 100:
                                if elbow > 160 and shoulder > 40 and hip > 160:
                                    self.feedback = "Up"
                                    if self.direction == 1:
                                        self.count += 0.5
                                        self.direction = 0
                                else:
                                    self.feedback = "Fix Form"
                                    self.error += 1
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                           b'Count: ' + str(self.count).encode() + b'\r\n'
                           b'Stage: ' + self.feedback.encode() + b'\r\n')

    def release_camera(self):
        self.camera.release()

pushup_tracker = PushupTracker()

# Initialize SquatTracker
class SquatTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.camera = cv2.VideoCapture(0)
        self.detector = PoseDetector()
        self.count = 0
        self.feedback = "Fixed Form"
        self.form = 0
        self.error = 0
        self.direction = 0

    def gen_frames(self):
        while True:
            success, frame = self.camera.read()
            if not success:
                break
            else:
                if frame is not None:
                    frame, results = self.detector.find_pose(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = self.pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.pose_landmarks:
                        lm_list = self.detector.find_position(frame, [], results)
                        right_leg = self.detector.find_angle(frame, lm_list, 24, 26, 28, results, draw=False)
                        left_leg = self.detector.find_angle(frame, lm_list, 23, 25, 27, results, draw=False)

                        if right_leg >= 150 and left_leg >= 150:
                            self.form = 1

                        if self.form == 1:
                                if right_leg >= 160 and left_leg >= 160:
                                    self.feedback = "Do Squat"
                                elif right_leg < 80 and left_leg < 80:
                                    self.error += 1
                                    self.form = 0
                                elif right_leg >= 80 and right_leg<=100 and left_leg >= 80 and left_leg<=100:
                                    self.feedback = "Correct Form. Keep Going"
                                    self.count += 1
                                    self.form = 0
                        else:
                            self.form = 0
                    
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                           b'Count: ' + str(self.count).encode() + b'\r\n'
                           b'Stage: ' + self.feedback.encode() + b'\r\n')

    def release_camera(self):
        self.camera.release()

squat_tracker = SquatTracker()

# Initialize BicepCurlTracker
class BicepCurlTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.camera = cv2.VideoCapture(0)
        self.detector = PoseDetector()
        self.count = 0
        self.stage = "Fixed Form"

    def calculate_angle(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = self.calculate_angle_helper(shoulder, elbow, wrist)
                return angle, self.stage
            else:
                return 0, self.stage
        except Exception as e:
            return 0, self.stage

    def calculate_angle_helper(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def gen_frames(self):
        while True:
            success, frame = self.camera.read()
            if not success:
                break
            else:
                if frame is not None:
                    frame, results = self.detector.find_pose(frame)
                    angle, self.stage = self.calculate_angle(frame)

                    if angle > 160:
                        self.stage = "down"
                    if angle < 30 and self.stage == "down":
                        self.stage = "up"
                        self.count += 1

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                           b'Count: ' + str(self.count).encode() + b'\r\n'
                           b'Stage: ' + self.stage.encode() + b'\r\n')

    def release_camera(self):
        self.camera.release()

bicep_curl_tracker = BicepCurlTracker()

@app.route('/pushup')
def pushup():
    # Reinitialize the pushup tracker
    global pushup_tracker
    pushup_tracker = PushupTracker()
    return render_template('pushup.html')

@app.route('/pushup_video_feed')
def pushup_video_feed():
    return Response(pushup_tracker.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pushup_data')
def pushup_data():
    return jsonify(count=pushup_tracker.count, feedback=pushup_tracker.feedback)

@app.route('/squat')
def squat():
    # Reinitialize the squat tracker
    global squat_tracker
    squat_tracker = SquatTracker()
    return render_template('squat.html')

@app.route('/squat_video_feed')
def squat_video_feed():
    return Response(squat_tracker.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squat_data')
def squat_data():
    return jsonify(count=squat_tracker.count, feedback=squat_tracker.feedback)

@app.route('/bicep')
def bicep():
    # Reinitialize the bicep curl tracker
    global bicep_curl_tracker
    bicep_curl_tracker = BicepCurlTracker()
    return render_template('bicep.html')

@app.route('/bicep_video_feed')
def bicep_video_feed():
    return Response(bicep_curl_tracker.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bicep_data')
def bicep_data():
    return jsonify(count=bicep_curl_tracker.count, stage=bicep_curl_tracker.stage)

@app.route('/')
def index():
    pushup_tracker.release_camera()
    squat_tracker.release_camera()
    bicep_curl_tracker.release_camera()
    return render_template('index.html')

@app.route('/stop_pushup_camera')
def stop_pushup_camera():
    pushup_tracker.release_camera()
    return '', 204

@app.route('/stop_squat_camera')
def stop_squat_camera():
    # Release the squat camera
    squat_tracker.release_camera()
    return '', 204

@app.route('/stop_bicep_camera')
def stop_bicep_camera():
    # Release the bicep curl camera
    bicep_curl_tracker.release_camera()
    return '', 204

@app.route('/user_info_pushup')
def user_info():
    return render_template('user_info.html', exercise_name="pushup", count=pushup_tracker.count, feedback=pushup_tracker.feedback)
@app.route('/user_info_squat')
def user_info_squat():
    return render_template('user_info.html', exercise_name="squat", count=squat_tracker.count, feedback=squat_tracker.feedback)
@app.route('/user_info_bicep')
def user_info_bicep():
    return render_template('user_info.html', exercise_name="bicep", count=bicep_curl_tracker.count, feedback=bicep_curl_tracker.stage)

def bicep():
    global bicep_curl_tracker
    bicep_curl_tracker = BicepCurlTracker()
    return render_template('bicep.html')

@app.route('/get_insights', methods=['POST'])
def get_insights():
    try:
        data = request.json
        print("step1")
        response = fetch_insights(data)
        print("step2")
        print(response)
        return jsonify(response=response), 200
    except Exception as e:
        print("Error")
        return jsonify(error=str(e)), 500

@app.route('/insights')
def show_insights():
    response = request.args.get('response')
    return render_template('insights.html', response=response)

def fetch_insights(data):
    genai.configure(api_key='')
    generation_config = {
  "temperature": 0.5,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 4058,
}
    
    safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

    model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    print('step3')
    data_json = json.dumps(data)
    feet=data['feet']
    inches=data['inches']
    weight=data['weight']
    print('stepmid')
    heightMeters = (feet * 12 + inches) * 0.0254
    bmi = weight / (heightMeters * heightMeters)
    print(bmi)
    print('step4')
    print(data_json)
    if feet<7 and feet>4 and weight<200 and weight>30:
        prompt_parts = [f"""Given a user's workout data in JSON format, which includes their name, exercise type, and duration, your task is to provide a comprehensive analysis and personalized recommendations. Begin by addressing the user by name and acknowledging their dedication to fitness.

Analyze the user's exercise routines, delving into specific techniques for improving performance and effectiveness. Incorporate real-world statistics or studies related to each exercise type, highlighting the benefits and potential pitfalls. Offer detailed advice in points for optimizing each workout session, considering factors like form, intensity, and rest intervals.

Next, discuss weight management strategies tailored to the user’s BMI and BMI category. Draw upon research findings or expert opinions to support your recommendations. Emphasize the importance of a balanced approach, encompassing both diet and exercise, in achieving and maintaining a healthy weight.

Incorporate suggestions for Indian food items (atleast 5 to 10) with their approximate Nutritional value that align with the user's BMI category, emphasizing nutrient-rich options that support overall health and fitness goals. Consider traditional ingredients and dishes known for their nutritional value and compatibility with different BMI ranges.

Conclude with an inspiring quote or anecdote to motivate the user to continue their fitness journey with enthusiasm and perseverance. Encourage them to embrace the journey towards a healthier lifestyle, celebrating progress and staying committed to their goals.

Remember to maintain a positive and supportive tone throughout the response, fostering a sense of empowerment and encouragement for the user to make positive changes in their fitness routine.
user_data: {data_json}
bmi: {bmi}"""]
    else:
        prompt_parts = [f""" username: {data['name']}
                            Just greet user and say that we are unable to provide insights as the data is not valid. Please enter valid data. """]
    response = model.generate_content(prompt_parts)
    text=response.text
    print(text)
    formatted_response = re.sub(r'\*\*|\*|_', '', text)
    print(formatted_response)
    return formatted_response

def get_completion(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.5,
    )
    return response.choices[0].message["content"]

if __name__ == "__main__":
    app.run(port=2000)

# Release cameras when the application stops
pushup_tracker.release_camera()
squat_tracker.release_camera()
bicep_curl_tracker.release_camera()
