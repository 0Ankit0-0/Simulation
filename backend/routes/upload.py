import os
from flask import Blueprint, request, jsonify
import werkzeug.utils as werkzeug_utils

upload_bp = Blueprint('upload', __name__)

# Fix: Use consistent path structure
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploaded_evidence')
UPLOAD_FOLDER = os.path.abspath(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'docx', 'png', 'txt', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: 
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = werkzeug_utils.secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    else:
        return jsonify({'error': 'Invalid file type', 'allowed_extensions': list(ALLOWED_EXTENSIONS)}), 400