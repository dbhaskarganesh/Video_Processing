import cv2
from flask import Flask, render_template, request, Response, redirect, url_for
import os

app = Flask(__name__)

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames(video_file):
    cap = cv2.VideoCapture(video_file)

    while True:
        success, frame = cap.read()

        if not success:
            break
        else:
            # Apply background subtraction to detect moving objects
            fgmask = fgbg.apply(frame)

            # Perform some morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw rectangles around the detected objects and print a message
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Moving Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = os.path.join('uploads', 'uploaded_video.mp4')
            file.save(filename)
            return redirect(url_for('video'))

    return Response(generate_frames('uploads/uploaded_video.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = os.path.join('uploads', 'uploaded_video.mp4')
        file.save(filename)
        return redirect(url_for('video'))

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
