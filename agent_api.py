from flask import Flask, request, jsonify
from Agent.data_agent import SalesDataAgent  # Assuming you have this

app = Flask(__name__)
agent = SalesDataAgent()

@app.route('/call-agent', methods=['POST'])
def call_agent():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Use the agent to process the prompt
    result = agent.run(prompt=prompt)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

