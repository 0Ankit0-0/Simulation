from flask import Flask, request, make_response
from flask_cors import CORS
from routes.upload import upload_bp
from routes.parse import parse_bp
from routes.case import case_bp
from routes.review import review_bp

app = Flask(__name__)

# GitHub Codespaces and development CORS configuration
CORS(app, 
    origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173", 
        "https://solid-invention-r4wwx5966wqpfx759-5173.app.github.dev",
        "https://*.app.github.dev",  # Allow all GitHub Codespaces domains
        "*"  # Allow all origins for development
    ],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type", 
        "Authorization", 
        "X-Requested-With",
        "Accept",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    supports_credentials=True,
    expose_headers=["Content-Range", "X-Content-Range"],
    max_age=3600  # Cache preflight for 1 hour
)

# Add explicit preflight handler for all routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin', '*'))
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response

# Add CORS headers to all responses
@app.after_request  
def after_request(response):
    origin = request.headers.get('Origin')
    if origin:
        response.headers.add('Access-Control-Allow-Origin', origin)
    else:
        response.headers.add('Access-Control-Allow-Origin', '*')
    
    response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,X-Requested-With")
    response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Routes
app.register_blueprint(upload_bp, url_prefix="/api")
app.register_blueprint(parse_bp, url_prefix="/api")
app.register_blueprint(case_bp, url_prefix="/api")
app.register_blueprint(review_bp, url_prefix="/api")

@app.route("/")
def home():
    return {"message": "Flask is running!, Nice to meet you!"}

# Test endpoint for CORS verification
@app.route("/api/test", methods=["GET", "POST", "OPTIONS"])
def test_cors():
    """Test endpoint to verify CORS is working"""
    return {
        "message": "CORS is working!",
        "method": request.method,
        "origin": request.headers.get('Origin', 'No origin header'),
        "headers": dict(request.headers)
    }, 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)