from flask import Flask, jsonify
import random
import time
from main import NeonAssistant

app = Flask(__name__)

assistant = NeonAssistant()


@app.route('/api/run_query', methods=['GET'])
def run_query():
    assistant.start_recording()
    return jsonify({})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
