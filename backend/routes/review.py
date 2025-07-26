import logging
from flask import Blueprint, request, jsonify
from bson.objectid import ObjectId
from datetime import datetime
from cerberus import Validator
from db.mongo import case_collection as cc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed blueprint name to avoid conflict with case.py
review_bp = Blueprint('review', __name__)

# Validation schemas
evidence_schema = {
    'filename': {'type': 'string', 'required': True},
    'approved': {'type': 'boolean', 'required': True},
    'reviewer_notes': {'type': 'string', 'required': False, 'maxlength': 1000},
    'reviewed_at': {'type': 'string', 'required': False}
}

review_schema = {
    'case_id': {'type': 'string', 'required': True, 'minlength': 10},
    'evidence': {'type': 'list', 'schema': {'type': 'dict', 'schema': evidence_schema}},
    'overall_status': {'type': 'string', 'allowed': ['approved', 'rejected', 'needs_revision'], 'required': False},
    'reviewer_id': {'type': 'string', 'required': False},
    'review_notes': {'type': 'string', 'required': False, 'maxlength': 2000}
}

def validate_review_data(data):
    """Validate review input data"""
    v = Validator(review_schema)
    return v.validate(data), v.errors

def validate_evidence_list(evidence_list):
    """Additional validation for evidence list"""
    if not evidence_list:
        return False, "Evidence list cannot be empty"
    
    if len(evidence_list) > 50:  # Reasonable limit
        return False, "Too many evidence items"
    
    # Check for duplicate filenames
    filenames = [item.get('filename', '') for item in evidence_list]
    if len(filenames) != len(set(filenames)):
        return False, "Duplicate filenames in evidence list"
    
    return True, "Valid evidence list"

@review_bp.route('/review_case', methods=['POST'])
def review_case():
    """Review and update case evidence with approval status"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data
        is_valid, validation_errors = validate_review_data(data)
        if not is_valid:
            logger.warning(f"Invalid review data: {validation_errors}")
            return jsonify({'error': 'Invalid input data', 'details': validation_errors}), 400
        
        case_id = data.get("case_id")
        evidence_list = data.get("evidence", [])
        overall_status = data.get("overall_status", "reviewed")
        reviewer_id = data.get("reviewer_id", "unknown")
        review_notes = data.get("review_notes", "")
        
        # Additional evidence validation
        is_evidence_valid, evidence_error = validate_evidence_list(evidence_list)
        if not is_evidence_valid:
            return jsonify({'error': evidence_error}), 400
        
        # Check if case exists
        existing_case = cc.find_one({"case_id": case_id})
        if not existing_case:
            logger.warning(f"Case not found for review: {case_id}")
            return jsonify({'error': 'Case not found'}), 404
        
        # Prepare updated evidence with review metadata
        updated_evidence = []
        approved_count = 0
        rejected_count = 0
        
        for evidence_item in evidence_list:
            # Add review timestamp
            evidence_item["reviewed_at"] = datetime.now().isoformat()
            evidence_item["reviewer_id"] = reviewer_id
            
            # Count approvals/rejections
            if evidence_item.get("approved") is True:
                approved_count += 1
            elif evidence_item.get("approved") is False:
                rejected_count += 1
            
            updated_evidence.append(evidence_item)
        
        # Prepare update document
        update_doc = {
            "evidence": updated_evidence,
            "reviewed": True,
            "reviewed_at": datetime.now().isoformat(),
            "reviewer_id": reviewer_id,
            "review_summary": {
                "total_evidence": len(updated_evidence),
                "approved_count": approved_count,
                "rejected_count": rejected_count,
                "pending_count": len(updated_evidence) - approved_count - rejected_count
            },
            "overall_status": overall_status,
            "review_notes": review_notes,
            "last_modified": datetime.now().isoformat()
        }
        
        # Update case in database
        result = cc.update_one(
            {"case_id": case_id},
            {"$set": update_doc}
        )
        
        if result.modified_count == 0:
            logger.error(f"Failed to update case: {case_id}")
            return jsonify({'error': 'Failed to update case'}), 500
        
        logger.info(f"Case reviewed successfully: {case_id} by {reviewer_id}")
        
        response_data = {
            "message": "Review saved successfully",
            "case_id": case_id,
            "review_summary": update_doc["review_summary"],
            "overall_status": overall_status
        }
        
        return jsonify(response_data), 200
    
    except ValueError as e:
        logger.error(f"Invalid input in review_case: {e}")
        return jsonify({'error': 'Invalid input data'}), 400
    
    except Exception as e:
        logger.error(f"Unexpected error in review_case: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@review_bp.route('/get_pending_reviews', methods=['GET'])
def get_pending_reviews():
    """Get list of cases pending review"""
    try:
        # Pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        
        # Calculate skip value for pagination
        skip = (page - 1) * per_page
        
        # Query for unreviewed cases
        pending_cases = list(cc.find(
            {"reviewed": {"$ne": True}},
            {"case_id": 1, "title": 1, "submitted_at": 1, "case_type": 1, "_id": 0}
        ).skip(skip).limit(per_page).sort("submitted_at", -1))
        
        # Get total count for pagination
        total_count = cc.count_documents({"reviewed": {"$ne": True}})
        
        response_data = {
            "pending_reviews": pending_cases,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total_count,
                "pages": (total_count + per_page - 1) // per_page
            }
        }
        
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.error(f"Error getting pending reviews: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@review_bp.route('/get_review_stats', methods=['GET'])
def get_review_stats():
    """Get review statistics"""
    try:
        # Aggregate statistics
        stats = {}
        
        # Total cases
        stats['total_cases'] = cc.count_documents({})
        
        # Reviewed vs pending
        stats['reviewed_cases'] = cc.count_documents({"reviewed": True})
        stats['pending_cases'] = cc.count_documents({"reviewed": {"$ne": True}})
        
        # Cases by status
        status_pipeline = [
            {"$match": {"reviewed": True}},
            {"$group": {"_id": "$overall_status", "count": {"$sum": 1}}}
        ]
        status_results = list(cc.aggregate(status_pipeline))
        stats['by_status'] = {item['_id']: item['count'] for item in status_results}
        
        # Recent review activity (last 7 days)
        from datetime import datetime, timedelta
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        stats['recent_reviews'] = cc.count_documents({
            "reviewed_at": {"$gte": week_ago}
        })
        
        return jsonify(stats), 200
    
    except Exception as e:
        logger.error(f"Error getting review stats: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@review_bp.route('/bulk_approve', methods=['POST'])
def bulk_approve_evidence():
    """Bulk approve evidence for a case"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        case_id = data.get('case_id')
        evidence_filenames = data.get('evidence_filenames', [])
        reviewer_id = data.get('reviewer_id', 'unknown')
        
        if not case_id:
            return jsonify({'error': 'Case ID is required'}), 400
        
        if not evidence_filenames:
            return jsonify({'error': 'Evidence filenames are required'}), 400
        
        # Check if case exists
        existing_case = cc.find_one({"case_id": case_id})
        if not existing_case:
            return jsonify({'error': 'Case not found'}), 404
        
        # Update specific evidence items
        current_time = datetime.now().isoformat()
        
        # Build update query to approve specific evidence
        update_result = cc.update_one(
            {"case_id": case_id},
            {
                "$set": {
                    "evidence.$[elem].approved": True,
                    "evidence.$[elem].reviewed_at": current_time,
                    "evidence.$[elem].reviewer_id": reviewer_id,
                    "last_modified": current_time
                }
            },
            array_filters=[{"elem.filename": {"$in": evidence_filenames}}]
        )
        
        if update_result.modified_count == 0:
            return jsonify({'error': 'No evidence was updated'}), 400
        
        logger.info(f"Bulk approved {len(evidence_filenames)} evidence items for case: {case_id}")
        
        return jsonify({
            "message": "Evidence approved successfully",
            "case_id": case_id,
            "approved_count": len(evidence_filenames)
        }), 200
    
    except Exception as e:
        logger.error(f"Error in bulk approve: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@review_bp.errorhandler(400)
def handle_bad_request(e):
    """Handle bad request errors"""
    return jsonify({'error': 'Bad request'}), 400

@review_bp.errorhandler(404)
def handle_not_found(e):
    """Handle not found errors"""
    return jsonify({'error': 'Resource not found'}), 404

@review_bp.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500