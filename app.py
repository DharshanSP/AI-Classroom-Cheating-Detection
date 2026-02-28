from flask import Flask, render_template, request
import os
from video_processor import analyze_video

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/snapshots", exist_ok=True)

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    video = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    result = analyze_video(video_path)

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)