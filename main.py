import cv2
from flask import Flask, render_template, Response, request
import face_recognition
import mysql.connector
import pandas as pd
import numpy as np

app = Flask(__name__)

connector = mysql.connector.connect(host="localhost", user="root", password="kempk2933g", database="employee")
cursor=connector.cursor()
# Function to retrieve data from the database
def get_data(cursor):
    cursor.execute("SELECT * FROM emp_data")
    result = cursor.fetchall()
    df = pd.DataFrame(result, columns=["Student_id", 'First_name', 'Last_name', 'Course', "Encoding"])
    return df

# Function to extract face encodings from the database data
def get_encoding(df):
    know_face_encoding = []
    know_face_name = []
    for i in df["Encoding"]:
        encoding_str = i[1:-1]  # Remove square brackets from the string
        encoding_float = [float(x) for x in encoding_str.split()]  # Convert string to list of floats
        know_face_encoding.append(np.array(encoding_float, dtype=float))
    
    know_face_name = list(df["First_name"])
    return know_face_encoding, know_face_name

df = get_data(cursor)
know_face_encoding, know_face_name = get_encoding(df)
# print("this is encoding ", know_face_encoding[0])
# print(type(know_face_encoding[0]))
# print("this is user name ", know_face_name[0])
# Function to process frames from the webcam stream
def gen_frames():
    cursor = connector.cursor()
    df = get_data(cursor)
    know_face_encoding, know_face_name = get_encoding(df)
    employees = know_face_name.copy()
    
    cap = cv2.VideoCapture(0)
    attend=[]
    while True:
        success, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(know_face_encoding, face_encoding)
            name = ''
            face_distances = face_recognition.face_distance(know_face_encoding, face_encoding)
            sorted_face_distances = sorted(face_distances)
            min_distance = sorted_face_distances[0] if sorted_face_distances else None

            if min_distance is not None and min_distance < 0.40:  # Adjust the threshold as needed
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = know_face_name[best_match_index]

            face_names.append(name)
            attend.append(name)
            # print(attend.count(name))
            
            if attend.count(name) == 1:
                cursor.execute("INSERT INTO attendance (name, entry_time) VALUES (%s, now())", (name,))
                connector.commit()

            if name in know_face_name and name in employees:
                employees.remove(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cursor.close()

# Route to render the HTML page with the video stream
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login_new():
    email = request.form['login_id']
    password = request.form['password']
    if email == "regex@123" and password == 'regex@123':
        return render_template('ragistration.html') 
    else:
        return render_template('index.html', error="Invalid Credentials")


@app.route('/ragis', methods=['POST'])
def new_func():
    return render_template("ragistration_new.html")


@app.route('/ragis_abc', methods=['POST'])
def submit():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    course =  request.form['course']
    photo = request.files['photo']  
    img_array = cv2.imdecode(np.fromstring(photo.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Detect face encodings
    encoding = face_recognition.face_encodings(img_array)
    if not encoding:  # Check if encoding list is empty
        return render_template("ragistration_new.html", error="No face detected. Please upload a valid image.")


    know_face_encoding.append(np.array(encoding))
    know_face_name.append(f"{first_name}' '{last_name}")
    enco = encoding[0]
    
    # print("this is encoding", np.array(enco))
    cursor.execute("insert into emp_data(first_name,last_name,Course,encoding) values('{}','{}','{}','{}')".format(first_name,last_name,course,enco))
    cursor.execute("commit")
    return render_template("ragistration_new.html", error="Registration Done")
    



# Route to provide the video stream
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')

