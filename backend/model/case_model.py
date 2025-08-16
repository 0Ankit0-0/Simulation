
from db.mongo import case_collection

def save_case(case_data):
    """Save a new case to the database"""
    if case_collection is None:
        raise ConnectionError("Database connection not available")
    try:
        result = case_collection.insert_one(case_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving case: {str(e)}")
        raise e


def get_case_by_id(case_id):
    """Retrieve a case by its case_id"""
    try:
        case = case_collection.find_one({"case_id": case_id})
        if case:
            # Convert ObjectId to string for JSON serialization
            case['_id'] = str(case['_id'])
        return case
    except Exception as e:
        print(f"Error retrieving case: {str(e)}")
        raise e

def update_case(case_id, update_data):
    """Update a case"""
    try:
        result = case_collection.update_one(
            {"case_id": case_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"Error updating case: {str(e)}")
        raise e