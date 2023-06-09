import os
from flask import request, jsonify,Flask
from werkzeug.datastructures import MultiDict
from tensorflow import keras
import json
import requests
from PIL import Image
import numpy as np



def download_file_from_google_drive(donwload_url, destination_path):
    response = requests.get(donwload_url, stream=True)

    if response.status_code == 200:
        with open(destination_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    else:
        return False

download_url = "https://download1477.mediafire.com/60ydhdofu25gBnh985LH7ji1LWRltRqx-HUtInpYeJoZagIjdr0zOrfB5686KI-iKep28a35n_SvgaB-TgiGOB7v_rCmyA0dISdk2PiQY1xxbw0socyz9ro9ZPeX5nOH2WoCd_bRLPtaCj7DgrzGkps0G7C-DoMtPYLW182wbkNE4Q/flutyxp1xc2op7o/PlantDiseaseDetectionModel.h5"  
destination_path = "PlantDiseaseDetectionModel.h5"

if not os.path.exists("PlantDiseaseDetectionModel.h5"):
    success = download_file_from_google_drive(download_url, destination_path)    
    if success:
        print("File downloaded successfully.")
    else:
        print("Error occurred while downloading the file.")



app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data  = MultiDict(request.form)
        img_file = request.files.get('image')
        if img_file:
            img_file.save(img_file.filename) # type: ignore
            # Preprocess the image
            image = Image.open(img_file.filename) # type: ignore

            # Resize the image to the target size
            image = image.resize((256, 256))

            # Convert the image to a numpy array
            image = np.array(image)

            # Normalize the image
            image = image / 255.0
            # Reshape the image to match the model's input shape
            image = image.reshape((1, 256, 256, 3))
            # Perform the prediction
            model = keras.models.load_model('PlantDiseaseDetectionModel.h5')
            predictions = model.predict(image) #type: ignore
            # Get the predicted class with the highest probability
            predicted_class = predictions.argmax(axis=1)
            probability = predictions[0][predicted_class[0]]
            with open('class_names.json', 'r') as classes_file:
                class_names = json.load(classes_file)
            
            message = {
                'Predicted Class': class_names[predicted_class[0]],
                'Probability': str(probability)
            }
            response = {
                'status': 200,
                'message': message
            }

            return jsonify(response)
        else:
            response = {
                'status': 400,
                'message': 'Image not found'
            }
            return jsonify(response)
    except Exception as e:
        response = {
            'status': 400,
            'message': str(e)
        }

    finally:
        file_path = img_file.filename #type: ignore
        if os.path.exists(file_path): #type: ignore
            os.remove(file_path) #type: ignore

    return jsonify(response)




if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000)) #type: ignore

