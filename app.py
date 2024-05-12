from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

#load your trained model
model = load_model('C:\\Users\\urbam\\Downloads\\chest_xray_classifier.h5')

@app.route('/',methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['file']

        if file and file.content_length > 0:  #check if a file has been sent
            #read the file into a memory stream
            filestream = io.bytesIO(file.read())

            #load the image from the bytesIO object
            img = load_img(filestream, target_size=(224,224))

            #preprocess the image as required by your model
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  #scale pixel values as your model aspects

            #make the prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            result = 'Pneumonia' if predicted_class == 1 else 'Normal'

            return render_template('index.html', result=result)
    

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)