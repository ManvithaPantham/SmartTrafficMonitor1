from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

VIDEO = "traffic.mp4"

cap = cv2.VideoCapture(VIDEO)

# vehicle class names
vehicle_names = {
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck"
}

# vehicle ID counter
vehicle_id = 0

def generate():

    global cap
    global vehicle_id

    while True:

        success, frame = cap.read()

        if not success:
            cap = cv2.VideoCapture(VIDEO)
            continue

        results = model(frame)

        vehicle_id = 0  # reset per frame

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                if cls in vehicle_names:

                    vehicle_id += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    name = vehicle_names[cls]

                    label = f"ID:{vehicle_id} {name}"

                    # draw rectangle
                    cv2.rectangle(frame,
                                  (x1, y1),
                                  (x2, y2),
                                  (0, 200, 0),
                                  2)

                    # draw label background
                    cv2.rectangle(frame,
                                  (x1, y1-30),
                                  (x1+160, y1),
                                  (0, 200, 0),
                                  -1)

                    # draw text
                    cv2.putText(frame,
                                label,
                                (x1+5, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes +
               b'\r\n')


@app.route("/video")
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/")
def home():
    return "Camera Server Running"


if __name__ == "__main__":
    app.run(port=5001, debug=True)