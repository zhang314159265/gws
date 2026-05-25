from flask import Flask, request

app = Flask(__name__)

@app.route("/v1/chat/completions", methods=["POST"])
def _():
    body = request.get_json()
    breakpoint()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000")
