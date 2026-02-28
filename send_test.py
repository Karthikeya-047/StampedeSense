import cv2
import requests

cap = cv2.VideoCapture("134437-759734827_medium.mp4")
url = "http://127.0.0.1:5000/process"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, img_encoded = cv2.imencode('.jpg', frame)

    response = requests.post(
        url,
        files={"image": img_encoded.tobytes()}
    )

    print(response.json())
