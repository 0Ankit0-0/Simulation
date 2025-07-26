from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'legal_case_db')

try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    case_collection = db['cases']
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    case_collection = None