import cv2
import dlib
import numpy as np
import os
from threading import Thread, Lock
from queue import Queue

def register_new_face(face_model, known_faces, known_names):
    """Capture a new face from the webcam, register it, and update known faces."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        # Display the frame to help user adjust their face position
        cv2.imshow("Register New Face - Press 's' to Save", frame)

        # Wait for the user to press 's' to save the image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            face_locations = face_model.shape_predictor(rgb_frame, face_location)
            if len(face_locations) == 0:
                print("No face detected. Please try again.")
                continue

            # Get the first face location
            face_location = face_locations[0]
            top, right, bottom, left = face_location.top(), face_location.right(), face_location.bottom(), face_location.left()

            # Get the landmarks and face encoding
            landmarks = face_model.shape_predictor(rgb_frame, face_location)
            face_descriptor = face_model.face_rec_model.compute_face_descriptor(rgb_frame, landmarks)
            face_encoding = np.array(face_descriptor)

            # Ask the user for their name
            name = input("Enter your name: ")

            # Create a folder for the new person
            person_folder = os.path.join("known_faces", name)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)

            # Save the captured face image
            image_path = os.path.join(person_folder, f"{name}.jpg")
            cv2.imwrite(image_path, frame)

            # Add the face encoding and name to the known faces list
            known_faces.append(face_encoding)
            known_names.append(name)

            print(f"Successfully registered {name}.")
            break

    cap.release()
    cv2.destroyAllWindows()


def load_known_faces(known_faces_dir):
    """Load known faces from the specified directory and encode them."""
    known_faces = []
    known_names = []

    # Load and process all images from the directory
    for entry in os.listdir(known_faces_dir):
        person_path = os.path.join(known_faces_dir, entry)
        if os.path.isdir(person_path):
            process_directory(person_path, entry, known_faces, known_names)
        elif os.path.isfile(person_path):
            process_file(person_path, entry, known_faces, known_names)

    return known_faces, known_names


def process_directory(person_path, person_name, known_faces, known_names):
    """Process a directory containing multiple images for a person."""
    for filename in os.listdir(person_path):
        image_path = os.path.join(person_path, filename)
        process_image(image_path, person_name, known_faces, known_names)


def process_file(image_path, filename, known_faces, known_names):
    """Process a single image file."""
    person_name = filename.split('.')[0]  # Assume file name is the person's name
    process_image(image_path, person_name, known_faces, known_names)


def process_image(image_path, person_name, known_faces, known_names):
    """Load and encode a single image."""
    image = dlib.load_rgb_image(image_path)
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    face_locations = detector(image)
    for face_location in face_locations:
        landmarks = shape_predictor(image, face_location)
        face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
        known_faces.append(np.array(face_descriptor))
        known_names.append(person_name)


class FaceModel:
    def __init__(self):
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


class FaceRecognitionThread(Thread):
    """Threaded class for handling face recognition in parallel."""
    def __init__(self, rgb_frame, face_locations, known_faces, known_names, face_model):
        Thread.__init__(self)
        self.rgb_frame = rgb_frame
        self.face_locations = face_locations
        self.known_faces = known_faces
        self.known_names = known_names
        self.face_model = face_model
        self.results = []

    def run(self):
        """Perform face recognition for detected faces."""
        for face_location in self.face_locations:
            landmarks = self.face_model.shape_predictor(self.rgb_frame, face_location)
            face_descriptor = self.face_model.face_rec_model.compute_face_descriptor(self.rgb_frame, landmarks)
            face_encoding = np.array(face_descriptor)

            # Compare detected face encoding to known faces
            if len(self.known_faces) > 0:
                matches = np.linalg.norm(self.known_faces - face_encoding, axis=1) < 0.6
                if True in matches:
                    match_index = np.argmin(np.linalg.norm(self.known_faces - face_encoding, axis=1))
                    name = self.known_names[match_index]
                    self.results.append((face_location, name))


def capture_and_display(known_faces, known_names, face_model):
    """Capture video from camera, perform face detection, tracking, and recognition."""
    cap = cv2.VideoCapture(0)
    frame_queue = Queue(maxsize=10)
    detection_interval = 10
    frame_count = 0

    def process_frames():
        while True:
            if not frame_queue.empty():
                rgb_frame, frame = frame_queue.get()
                if frame_count % detection_interval == 0:
                    face_locations = detect_faces(rgb_frame)
                    recognition_thread = FaceRecognitionThread(rgb_frame, face_locations, known_faces, known_names, face_model)
                    recognition_thread.start()
                    recognition_thread.join()

                    for face_location, name in recognition_thread.results:
                        top, right, bottom, left = face_location.top(), face_location.right(), face_location.bottom(), face_location.left()
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("Facial Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    processing_thread = Thread(target=process_frames)
    processing_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not frame_queue.full():
            frame_queue.put((rgb_frame, frame))

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def detect_faces(rgb_frame):
    """Detect faces in the provided frame."""
    detector = dlib.get_frontal_face_detector()
    return detector(rgb_frame)


# === Main Program ===
if __name__ == "__main__":
    # Load known faces
    known_faces, known_names = load_known_faces('known_faces')

    # Initialize models for face recognition
    face_model = FaceModel()

    # Option for the user to register a new face
    print("Do you want to register a new face? (y/n)")
    if input().lower() == 'y':
        register_new_face(face_model, known_faces, known_names)

    # After registration or skipping, proceed with facial recognition
    print("Face registration complete. Now proceeding with facial recognition.")
    
    # Start the facial recognition and video display process
    capture_and_display(known_faces, known_names, face_model)