print("Creating assistant..")  # on top for debug
import os
from flask import Flask, jsonify
import random
import time
from main import NeonAssistant

app = Flask(__name__)
assistant = NeonAssistant()
print("Ready")


@app.route('/api/run_query', methods=['GET'])
def run_query():
    print("Assistant listening")
    assistant.start_recording()
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5372, debug=False)
