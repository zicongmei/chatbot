from flask import Flask, request, jsonify
import subprocess
import requests

app = Flask(__name__)

def generate_response(prompt):
    """
    Generate a response by calling the Deepseek model through Ollama.
    Make sure that Ollama is installed and Deepseek is set up locally.
    """
    try:
        r = requests.post('http://localhost:11434/api/chat', json={
            "model": "deepseek-r1:14b",
            "messages": [
                {
                "role": "user",
                "content": prompt
                }
            ],
            "stream": False
            })
        # if r.status_code != 200:
        response = r.json()['message']['content']
    except Exception as e:
        response = f"Error generating response: {e}"

    return response

@app.route("/chat", methods=["POST"])
def chat():
    """
    API endpoint to receive a user message and return a chatbot response.
    """
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response_text = generate_response(user_message)
    return jsonify({"response": response_text})

# Optionally, serve the frontend from the Flask app
@app.route("/")
def home():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(debug=True)
