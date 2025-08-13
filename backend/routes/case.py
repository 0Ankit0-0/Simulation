import os
import uuid
import logging
from flask import Blueprint, request, jsonify
from datetime import datetime
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
from cerberus import Validator
from flask_cors import cross_origin

from services.parser_ai import parse_evidence
from model.case_model import save_case, get_case_by_id

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "evidence_uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "jpg", "jpeg", "png"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

case_bp = Blueprint("case", __name__)

# Validation schema for input
case_schema = {
    "title": {"type": "string", "minlength": 3, "maxlength": 200, "required": True},
    "description": {
        "type": "string",
        "minlength": 10,
        "maxlength": 2000,
        "required": True,
    },
    "case_type": {
        "type": "string",
        "allowed": [
            "civil",
            "criminal",
            "family",
            "corporate",
            "constitutional",
        ],  # ✅ Added
        "required": True,
    },
}


def validate_case_data(data):
    """Validate case input data"""
    v = Validator(case_schema)
    return v.validate(data), v.errors


def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file(file):
    """Comprehensive file validation"""
    if not file or file.filename == "":
        return False, "No file selected"

    if not allowed_file(file.filename):
        return False, f"File type not allowed. Allowed: {list(ALLOWED_EXTENSIONS)}"

    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB"

    if file_size == 0:
        return False, "File is empty"

    return True, "Valid file"


def generate_secure_filename(original_filename, case_id):
    """Generate secure unique filename"""
    if not original_filename:
        return None

    try:
        ext = secure_filename(original_filename).rsplit(".", 1)[1].lower()
        unique_id = str(uuid.uuid4())[:8]
        return f"{case_id}_{unique_id}.{ext}"
    except IndexError:
        return None


def ensure_upload_directory():
    """Ensure upload directory exists"""
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create upload directory: {e}")
        return False


@case_bp.route("/submit_case", methods=["POST", "OPTIONS"])
@cross_origin("https://solid-invention-r4wwx5966wqpfx759-5173.app.github.dev", supports_credentials=True)
def submit_case():
    """Submit a new case with evidence files"""
    try:
        # Extract form data
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        case_type = request.form.get("case_type", "").strip()
        files = request.files.getlist("evidence")

        # Validate input data
        case_data_input = {
            "title": title,
            "description": description,
            "case_type": case_type,
        }

        logger.info(f"Received form: {case_data_input}")
        logger.info(f"Received files: {[f.filename for f in files]}")

        is_valid, validation_errors = validate_case_data(case_data_input)
        if not is_valid:
            logger.warning(f"Invalid case data: {validation_errors}")
            return (
                jsonify({"error": "Invalid input data", "details": validation_errors}),
                400,
            )

        # Check if files are provided
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "At least one evidence file is required"}), 400

        # Ensure upload directory exists
        if not ensure_upload_directory():
            return jsonify({"error": "Server configuration error"}), 500

        # Generate unique case ID
        case_id = (
            f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
        )
        saved_evidence = []
        failed_files = []

        # Process each file
        for file in files:
            try:
                # Validate file
                is_valid_file, file_error = validate_file(file)
                if not is_valid_file:
                    failed_files.append(
                        {"filename": file.filename, "error": file_error}
                    )
                    continue

                # Generate secure filename
                secure_name = generate_secure_filename(file.filename, case_id)
                if not secure_name:
                    failed_files.append(
                        {"filename": file.filename, "error": "Invalid filename"}
                    )
                    continue

                # Save file
                filepath = os.path.join(UPLOAD_FOLDER, secure_name)
                file.save(filepath)

                # Parse evidence
                try:
                    parsed = parse_evidence(filepath)
                    parsed["approved"] = None
                    parsed["original_filename"] = file.filename
                    parsed["uploaded_at"] = datetime.now().isoformat()
                    saved_evidence.append(parsed)

                    logger.info(
                        f"Successfully processed file: {file.filename} for case: {case_id}"
                    )

                except Exception as parse_error:
                    logger.error(
                        f"Failed to parse evidence file {file.filename}: {parse_error}"
                    )
                    # Clean up file if parsing failed
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    failed_files.append(
                        {"filename": file.filename, "error": "Failed to process file"}
                    )

            except Exception as file_error:
                logger.error(f"Error processing file {file.filename}: {file_error}")
                failed_files.append(
                    {"filename": file.filename, "error": str(file_error)}
                )

        # Check if any files were successfully processed
        if not saved_evidence:
            return (
                jsonify(
                    {
                        "error": "No files could be processed successfully",
                        "failed_files": failed_files,
                    }
                ),
                400,
            )

        # Prepare case data for database
        case_data = {
            "case_id": case_id,
            "title": title,
            "description": description,
            "case_type": case_type,
            "submitted_at": datetime.now().isoformat(),
            "evidence": saved_evidence,
            "status": "submitted",
            "reviewed": False,
        }

        # Save to database
        save_case(case_data)

        response_data = {
            "message": "Case submitted successfully",
            "case_id": case_id,
            "processed_files": len(saved_evidence),
        }

        if failed_files:
            response_data["failed_files"] = failed_files
            response_data["warning"] = (
                f"{len(failed_files)} files could not be processed"
            )

        logger.info(f"Case submitted successfully: {case_id}")
        return jsonify(response_data), 200

    except ValueError as e:
        logger.error(f"Invalid input in submit_case: {e}")
        return jsonify({"error": "Invalid input data"}), 400

    except OSError as e:
        logger.error(f"File system error in submit_case: {e}")
        return jsonify({"error": "File processing error"}), 500

    except Exception as e:
        logger.error(f"Unexpected error in submit_case: {e}")
        return jsonify({"error": "Internal server error"}), 500


@case_bp.route("/get_case/<case_id>", methods=["GET"])
def get_case(case_id):
    """Retrieve case by ID"""
    try:
        # Basic input validation
        if not case_id or not isinstance(case_id, str):
            return jsonify({"error": "Invalid case ID"}), 400

        case_data = get_case_by_id(case_id)
        if case_data:
            # Remove sensitive fields if needed
            case_data.pop("_id", None)  # Remove MongoDB ObjectId
            logger.info(f"Case retrieved successfully: {case_id}")
            return jsonify(case_data), 200
        else:
            logger.warning(f"Case not found: {case_id}")
            return jsonify({"error": "Case not found"}), 404

    except Exception as e:
        logger.error(f"Error retrieving case {case_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@case_bp.route("/get_cases", methods=["GET"])
def get_cases():
    """Get list of all cases with pagination"""
    try:
        # Get pagination parameters
        page = request.args.get("page", 1, type=int)
        per_page = min(
            request.args.get("per_page", 10, type=int), 100
        )  # Max 100 per page
        case_type_filter = request.args.get("case_type", None)

        # This would need to be implemented in case_model
        # cases, total = get_cases_paginated(page, per_page, case_type_filter)

        # Placeholder response
        return (
            jsonify(
                {
                    "message": "Endpoint implemented but requires database method",
                    "page": page,
                    "per_page": per_page,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error retrieving cases: {e}")
        return jsonify({"error": "Internal server error"}), 500


@case_bp.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return (
        jsonify(
            {"error": f"File too large. Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB"}
        ),
        413,
    )
