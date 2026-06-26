from flask import Flask, jsonify
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb://localhost:27017/")
db = client["trafficDB"]
collection = db["vehicle_count"]

@app.route("/")
def home():
    return "Server Running"

@app.route("/data")
def get_data():

    data = list(collection.find({},{"_id":0}))

    return jsonify(data)

app.run(port=5000,debug=True)