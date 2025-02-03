from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import os
import numpy as np
from capture import detect_faces, FaceModel, load_known_faces
from threading import Thread, Lock
from queue import Queue
import logging
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a strong secret key

# Configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Dummy user class for demonstration
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Default user credentials
DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'password'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

# === Load Known Faces and Face Model ===
face_model = FaceModel()  # Initialize the face recognition model
known_faces, known_names = load_known_faces('known_faces')  # Load faces from directory

# Store the last detected unknown face encoding for registration
last_detected_face = None
last_detected_frame = None

# Queue to hold the latest frame for streaming
frame_queue = Queue(maxsize=1)

# List to hold detected names for updating the table
detected_names = []
detected_names_lock = Lock()

# Set to keep track of already displayed names
displayed_names = set()

# === Route for Home Page ===
@app.route('/')
@login_required
def index():
    """Home page with video stream and controls."""
    return render_template('index.html')

# === Login Route ===
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check credentials against default values
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            user = User(id=1)
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials', 'error')
    return render_template('login.html')

# === Logout Route ===
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# === Video Stream for Facial Recognition ===
def gen_video():
    """Generate video stream for facial recognition."""
    global last_detected_face, last_detected_frame

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend for better compatibility

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to a smaller resolution, e.g., 320x240
        frame = cv2.resize(frame, (350, 240))  # Adjust the size here

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = detect_faces(rgb_frame)

        for face_location in face_locations:
            top, right, bottom, left = face_location.top(), face_location.right(), face_location.bottom(), face_location.left()
            landmarks = face_model.shape_predictor(rgb_frame, face_location)
            face_descriptor = face_model.face_rec_model.compute_face_descriptor(rgb_frame, landmarks)
            face_encoding = np.array(face_descriptor)

            # Compare the face with known faces
            if len(known_faces) > 0:
                distances = np.linalg.norm(known_faces - face_encoding, axis=1)
                min_distance = np.min(distances)
                if min_distance < 0.6:
                    match_index = np.argmin(distances)
                    name = known_names[match_index]
                    confidence = 1 - min_distance  # Confidence score
                else:
                    name = "Unknown"
                    last_detected_face = face_encoding  # Save the unknown face encoding for registration
                    last_detected_frame = frame.copy()  # Save the frame for registration
            else:
                name = "Unknown"
                last_detected_face = face_encoding  # Save the unknown face encoding for registration
                last_detected_frame = frame.copy()  # Save the frame for registration

            # Draw a box around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Update detected names list if the face is recognized
            if name != "Unknown":
                with detected_names_lock:
                    detected_names.append((name, min_distance))

        # Encode the frame and stream it as a response
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
@login_required
def video_feed():
    """Route to stream the video feed."""
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === Route to Register a New Face ===
@app.route('/register', methods=['POST'])
@login_required
def register_face():
    """Route to register the last detected unknown face as a new face."""
    global last_detected_face, last_detected_frame

    name = request.form['name']  # Get the name entered in the form

    if last_detected_face is not None and last_detected_frame is not None:
        # Create a folder for the new person
        person_folder = os.path.join("known_faces", name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        # Save the last detected face image
        image_path = os.path.join(person_folder, f"{name}.jpg")
        cv2.imwrite(image_path, last_detected_frame)

        # Save the last detected face and update the known faces list
        known_faces.append(last_detected_face)
        known_names.append(name)

        # Reset last detected face and frame after registration
        last_detected_face = None
        last_detected_frame = None

        flash(f"Successfully registered {name}.", "success")
    else:
        flash("No face detected to register.", "error")

    # Redirect to the homepage
    return redirect(url_for('index'))

@app.route('/detected_names')
@login_required
def get_detected_names():
    """Route to get the list of detected names."""
    with detected_names_lock:
        names = list(detected_names)
        detected_names.clear()  # Clear the list after sending the names
    return jsonify(names=names)

if __name__ == "__main__":
    # Start video capture in background thread
    capture_thread = Thread(target=gen_video, daemon=True)
    capture_thread.start()
    # Start Flask app
    app.run(debug=True)