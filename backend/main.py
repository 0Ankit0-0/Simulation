from flask import Flask
from flask_cors import CORS
from routes.upload import upload_bp
from routes.parse import parse_bp
from routes.case import case_bp
from routes.review import review_bp

app = Flask(__name__)

# Configure CORS properly
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# Routes using blueprints
app.register_blueprint(upload_bp, url_prefix="/api")
app.register_blueprint(parse_bp, url_prefix="/api")
app.register_blueprint(case_bp, url_prefix="/api")
app.register_blueprint(review_bp, url_prefix="/api")

@app.route("/")
def home():
    return {"message": "Flask is running!, Nice to meet you!"}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)