from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection - use correct environment variable names
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = os.getenv('MONGO_DB', 'legal_case_db')  # Use MONGO_DB from your .env

try:
    client = MongoClient(MONGO_URI, 
                        serverSelectionTimeoutMS=5000,  # 5 second timeout
                        connectTimeoutMS=10000,         # 10 second connection timeout
                        socketTimeoutMS=10000)          # 10 second socket timeout
    
    # Test the connection
    client.admin.command('ping')
    
    db = client[DATABASE_NAME]
    case_collection = db['cases']
    print(f"Connected to MongoDB successfully - Database: {DATABASE_NAME}")
    
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    case_collection = None
    client = None
    db = None