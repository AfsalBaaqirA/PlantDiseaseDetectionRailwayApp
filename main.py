import os
from flask import request, jsonify,Flask
from werkzeug.datastructures import MultiDict
from tensorflow import keras
import json

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
            image = keras.preprocessing.image.load_img(img_file.filename, target_size=(256, 256))
            image = keras.preprocessing.image.img_to_array(image)
            image = image / 255.0  # Normalize the image
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
