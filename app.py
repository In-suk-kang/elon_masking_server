from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import func

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = func.image_processing()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            doc_processor = func.image_processing(file_path) 
            result_path = doc_processor.process_document()

            return send_file(result_path, as_attachment=True, mimetype='image/png', download_name='result_image.png')

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
