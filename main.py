from ultralytics import YOLO
import cv2
from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["trafficDB"]
collection = db["vehicle_count"]

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("traffic.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        cap = cv2.VideoCapture("traffic.mp4")
        continue

    results = model(frame)

    cars = 0
    bikes = 0
    buses = 0
    trucks = 0

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            # class IDs
            if cls == 2:
                cars += 1
                label = "Car"

            elif cls == 3:
                bikes += 1
                label = "Bike"

            elif cls == 5:
                buses += 1
                label = "Bus"

            elif cls == 7:
                trucks += 1
                label = "Truck"

            else:
                continue

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    total = cars + bikes + buses + trucks

    # Save to MongoDB
    data = {
        "vehicle_count": total,
        "cars": cars,
        "bikes": bikes,
        "buses": buses,
        "trucks": trucks,
        "time": datetime.now()
    }

    collection.insert_one(data)

    cv2.putText(frame,f"Total: {total}",
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,0,255),2)

    cv2.imshow("Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()