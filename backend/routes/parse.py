from flask import Blueprint, request, jsonify
from services.parser import extract_text_from_file as extract_text
import os

parse_bp = Blueprint('parse', __name__)

@parse_bp.route('/extract', methods=['POST'])
def parse_file():
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'Filename is required'}), 400
    
    # Fix: Use consistent path structure
    path = os.path.join(os.path.dirname(__file__), '..', 'uploaded_evidence', filename)
    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    
    text = extract_text(path)
    return jsonify({'text': text}), 200