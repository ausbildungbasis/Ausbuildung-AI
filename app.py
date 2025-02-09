from flask import Flask, request, jsonify
import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a model that supports German
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

DB_CONFIG = {
    "host": "test.ausbildungsbasis.de",
    "user": "uaixkdmalwgpa",
    "password": "Ausbildungsbasis123?.",
    "database": "dbbjv8sgihuufp"
}

def connect_db():
    """Establish MySQL database connection."""
    return mysql.connector.connect(**DB_CONFIG)

def load_candidates():
    """Fetch candidate profiles from MySQL database and format them properly."""
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            t.id, t.name, t.firstname, t.description, t.location, t.industrys, 
            GROUP_CONCAT(s.name SEPARATOR ', ') AS skills,
            GROUP_CONCAT(DISTINCT CONCAT(je.jobtitle, ' bei ', je.companyname) SEPARATOR ', ') AS experiences,
            GROUP_CONCAT(DISTINCT CONCAT(sc.degree, ' von ', sc.schoolname) SEPARATOR ', ') AS education
        FROM trainees AS t
        LEFT JOIN trainee_skills AS ts ON ts.trainee_id = t.id
        LEFT JOIN skills AS s ON ts.skill_id = s.id  
        LEFT JOIN job_experiences AS je ON je.trainee_id = t.id
        LEFT JOIN school_careers AS sc ON sc.trainee_id = t.id 
        GROUP BY t.id;
    """)
    candidates = cursor.fetchall()
    conn.close()  # Close the connection
    
    if not candidates:
        return []

    processed_candidates = []
    for candidate in candidates:
        full_text = (
             f"Beschreibung: {candidate['description']}\n"
             f"Fähigkeiten: {candidate['skills']}\n"
             f"Erfahrung: {candidate['experiences']}\n"
             f"Ausbildung: {candidate['education']}"
    )
        processed_candidates.append({
            "id": candidate["id"],
            "name": f"{candidate['firstname']} {candidate['name']}",
            "text": full_text
        })
    return processed_candidates  # Return properly formatted candidates

def rank_candidates(job_description, candidates):
    """Rank candidates based on similarity to job description."""
    # Encode the job description
    job_embedding = model.encode(job_description)
    # Encode all candidate texts
    candidate_embeddings = np.array([model.encode(candidate["text"]) for candidate in candidates])
    # Compute cosine similarity
    similarity_scores = cosine_similarity([job_embedding], candidate_embeddings)[0]
    
    # Attach similarity scores to candidates
    for i, candidate in enumerate(candidates):
        candidate["similarity"] = round(float(similarity_scores[i]), 2)

    # Return candidates sorted by similarity (highest first)
    return sorted(candidates, key=lambda x: x["similarity"], reverse=True)

# Create the Flask application
app = Flask(__name__)

@app.route('/rank_candidates', methods=['POST'])
def rank_candidates_api():

    data = request.get_json()
    if not data or "job_description" not in data:
        return jsonify({"error": "job_description is required"}), 400
    
    job_description = data["job_description"]
    
    # Load candidates from the database
    candidates = load_candidates()
    if not candidates:
        return jsonify({"error": "No candidates found"}), 404
    
    # Rank candidates based on the job description
    ranked_candidates = rank_candidates(job_description, candidates)
    
    # Optionally, you can limit the number of returned candidates
    return jsonify({"ranked_candidates": ranked_candidates[:10]}), 200

if __name__ == "__main__":
    # Run the Flask server on port 5000
    app.run(debug=True, host="0.0.0.0", port=5000)
