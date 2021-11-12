import random
import os
from flask import Flask, request, render_template
from speech2text_service import Speech2Text_Service

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def predict():
    transcript = ""
    if request.method == "POST":
        # GET THE AUDIO FILE AND SAVE IT
        audio_file = request.files['file']
        file_name = str(random.randint(0, 100000))
        audio_file.save(file_name)

        # invoke service
        s2t = Speech2Text_Service()

        # make prediction
        transcript = s2t.transcribe(file_name)

        # remove the audio files
        os.remove(file_name)
    return render_template('index.html', transcript=transcript)

if __name__ == "__main__":
    app.run(debug=True)