from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/check", methods=["POST"])
def check():
    data = request.json
    ideal = data["ideal"]
    student = data["student"]

    v1 = model.encode(ideal)
    v2 = model.encode(student)

    score = float(cosine_similarity([v1], [v2])[0][0]) * 100

    return jsonify({
        "score": round(score, 2),
        "result": "صحيح" if score >= 70 else "غلط"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
