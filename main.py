from datetime import date, datetime
from flask import Flask, jsonify, render_template, request, send_file, session
from io import BytesIO
import numpy as np
import os
import pandas as pd
import random
import string
from tqdm import tqdm

from PIL import Image
from cv2 import COLOR_GRAY2BGR, cvtColor, hconcat, resize, vconcat
import matplotlib.cm as cm

from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
import io



model = load_skin_model()




app = Flask(__name__)
app.secret_key = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/lab')
def lab():
    return render_template('lab.html')

@app.route('/test')
def test():
    return render_template('test.html')


def load_skin_model():
    return load_model('chest_disease_001--3.982547--0.467237--0.681417.h5', compile=False)
def get_image(file_path):
    img_path = file_path
    img = Image.open(img_path).convert('RGB').resize((300,300))
    img = np.asarray(img)
    img = tf.cast(img, tf.float32)
    img = np.expand_dims(img, axis=0)
    return img

def make_gradcam_heatmap(img_path, model, last_conv_layer_name, pred_index=None):
    im = get_image(img_path)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(im)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
            # class_channel = preds[3]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap_conv, alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img_array = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap_conv)
    jet = cm.get_cmap("jet", lut=256)  # Update this line
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = np.expand_dims(jet_heatmap, axis=0)  # Add an extra dimension
    jet_heatmap = jet_heatmap[0]  # Remove the extra dimension
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img_array

    # Concatenate original and superimposed images
    out_img = hconcat([img_array, superimposed_img])

    superimposed_img = keras.preprocessing.image.array_to_img(out_img)

    if not os.path.exists('gradcam_outputs/'):
        os.mkdir('gradcam_outputs/')
    cam_path = 'gradcam_outputs/gradcam_' + os.path.basename(img_path)
    

    session['cam_path'] = cam_path
    # Save the superimposed image
    superimposed_img.save(cam_path)



def predict(image_path, gradcam = True):
    disease = []
    # gradcam = True
    # image_path = 'tb0004.png'
    global model
    prediction = model.predict(get_image(image_path))
    if gradcam:
        heatmap_conv = make_gradcam_heatmap(image_path, model, 'top_conv')
        heatmap_multiply = make_gradcam_heatmap(image_path, model, 'multiply')
        save_and_display_gradcam(image_path, heatmap_conv)
    if prediction[0][0] > .5:
        disease = 'bacterial_pneumonia'
    if prediction[0][1] > .5:
        disease = 'covid'
    if prediction[0][2] > .5:
        disease = 'lung_opacity'
    if prediction[0][3] > .5:
        disease = 'normal'
    if prediction[0][4] > .3:
        disease = 'tuberculosis'
    if prediction[0][5] > .5:
        disease = 'viral_pneumonia'
    print(disease)
    # max_value = max(prediction[0])
    max_value = np.max(prediction[0])

    print(prediction)
    return disease, max_value




def save_image(image_data, filename = 'temp_uploaded_image.jpg'):
    with open(filename, 'wb') as f:
        f.write(image_data)


@app.route('/process_image', methods=['POST'])
def process_image():
    try:

        name = request.form['name']
        gender = request.form['gender']
        phone = request.form['phone']

        # store in session
        session['name'] = name
        session['gender'] = gender
        session['phone'] = phone
        
        # Get image data from the request
        image_data = request.files['image'].read()
        # get image name
        # image_name = request.files['image'].filename
        print("Reached here")

        # Save the image temporarily genrate it with name of patient
        image_name = name + '.jpg'
        temp_image_path = image_name
        
        
        save_image(image_data, temp_image_path)
        print("image saved")
       

        # Process the image
        # prediction, confidence = process_single_image(temp_image_path)
        # filename = image_data.filename
        disease, confidence = predict(image_name)
        print("result is ", disease, confidence  )

        # Delete the tmep image if it is in directory
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

       
        
        # Convert float32 to a JSON-serializable format
        confidence = float(confidence)

        return jsonify({
            'success': True,
            'message': 'Image processed successfully',
            'DiseaseName': disease,
            'Confidence': confidence
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})




@app.route('/save-graph', methods=['POST'])
def save_graph():
    name = session['name']
    gender = session['gender']
    phone = session['phone']
    # Generate a random id for the patient
    patient_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # Get the current date and time
    current_datetime = datetime.now()

    # Convert the current date to a string
    current_date_string = current_datetime.strftime("%Y-%m-%d")


    # Receive the graph image from the request and save it
    graph_image = request.files['graphImage']

    # generate random name for graph with patient name and its id
    graph_name = name + patient_id + '.png'
    graph_image.save(graph_name)

    # Open the image
    img = Image.open(graph_name)

    # Create a new image with a white background
    new_img = Image.new("RGB", img.size, "white")

    # Paste the original image onto the new image, preserving transparency if any
    new_img.paste(img, (0, 0), img)

    # Overwrite the original image with the new image
    new_img.save(graph_name)

    # get the path of the grad_cam image path
    cam_path = session['cam_path']


    # Define the paths to the existing PDF and images
    existing_pdf_path = 'template.pdf'
    image1_path = graph_name  # This is the graph image saved earlier
    image2_path = cam_path  # Another image to be added



    # Create a PDF file reader for the existing PDF
    existing_pdf = PdfReader(existing_pdf_path)

    # Load the first page of the existing PDF
    page = existing_pdf.pages[0]

    # Create ReportLab canvas for drawing
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)

    name = session['name']
    gender = session['gender']
    phone = session['phone']
    # Generate a random id for the patient
    patient_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # Get the current date and time
    current_datetime = datetime.now()

    # Convert the current date to a string
    current_date_string = current_datetime.strftime("%Y-%m-%d")
    print(type(current_date_string))


    # Set font size and color
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0, 0, 0)  # Black color
    # Modify the code to include text annotations
    c.drawString(120, 706, name)  # Coordinates for the Name
    c.drawString(99, 693, patient_id )  # Coordinates for the Patient ID
    c.drawString(470, 705.5, gender)  # Coordinates for Gender
    c.drawString(467, 693, phone)  # Coordinates for Contact
    c.drawString(495, 741, current_date_string)  # Coordinates for Date
    c.drawString(74, 254, "Negative")  # Coordinates for Result
    

    # # Adjust coordinates and dimensions as needed
    c.drawImage(image1_path, 50, 500, width=500, height=150)  # Coordinates for the first image
    c.drawImage(image2_path, 50, 300, width=500, height=150)  # Coordinates for the second image
    c.showPage()
    c.save()

    # Move the overlay canvas to a PDF file reader
    packet.seek(0)
    overlay_pdf = PdfReader(packet)

    # Create a PDF file writer for the output PDF
    output_pdf = PdfWriter()

    # Merge the canvas content with the existing PDF page
    page.merge_page(overlay_pdf.pages[0])

    # Add the modified page to the output PDF
    output_pdf.add_page(page)

    # Save the modified PDF to a file (or return it as a response)
    # modified_pdf_path = 'modified.pdf'
    # with open(modified_pdf_path, 'wb') as output_file:
    #     output_pdf.write(output_file)

    # # Return the modified PDF as a response
    # return send_file(modified_pdf_path, mimetype='application/pdf', as_attachment=False)
    # Save the modified PDF to a BytesIO object
    modified_pdf_bytes = BytesIO()
    output_pdf.write(modified_pdf_bytes)

     # Set the BytesIO object's position to the beginning
    modified_pdf_bytes.seek(0)


    # Return the modified PDF as a response with appropriate headers
    return send_file(
        modified_pdf_bytes,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='modified.pdf'
    )








if __name__ == '__main__':
    app.run(debug=True)
