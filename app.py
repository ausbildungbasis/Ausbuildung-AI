from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK requirements
nltk.download('punkt')
nltk.download('punkt_tab')

# Use the German Stemmer
stemmer = SnowballStemmer("german")

# MySQL Configuration
DB_CONFIG = {
    "host": "test.ausbildungsbasis.de",
    "user": "uaixkdmalwgpa",
    "password": "Ausbildungsbasis123?.",
    "database": "dbbjv8sgihuufp"
}

# Connect to MySQL
def connect_db():
    return mysql.connector.connect(**DB_CONFIG)

# Fetch job description based on job_id
def fetch_job_description(job_id):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(f"SELECT CONCAT(jobtitle, ' ', description) AS job_info FROM joblistings WHERE id = {job_id};")
    job = cursor.fetchone()
    conn.close()

    if not job:
        return None

    return job["job_info"]

# Fetch and process candidate data
def load_candidates():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            t.id, t.name, t.firstname, t.description, 
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
    conn.close()

    if not candidates:
        return []

    processed_candidates = []
    for candidate in candidates:
        processed_candidates.append({
            "id": candidate["id"],
            "name": f"{candidate['firstname']} {candidate['name']}",
            "description": candidate["description"] if candidate["description"] else "",
            "skills": candidate["skills"] if candidate["skills"] else "",
            "experiences": candidate["experiences"] if candidate["experiences"] else "",
            "education": candidate["education"] if candidate["education"] else ""
        })
    return processed_candidates

# Text preprocessing: Tokenization + Stemming
def preprocess_text(text):
    words = word_tokenize(text.lower(), language="german")
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

# Rank candidates using TF-IDF
def rank_candidates(job_description, candidates):
    texts = [preprocess_text(job_description)] + [preprocess_text(candidate["description"]) for candidate in candidates]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

    for i, candidate in enumerate(candidates):
        candidate["similarity"] = f"{round(float(similarity_scores[i]) * 100, 2)}%"

    return sorted(candidates, key=lambda x: x["similarity"], reverse=True)

# Flask API
app = Flask(__name__)
CORS(app)

@app.route('/<int:job_id>', methods=['GET'])
def rank_candidates_api(job_id):
    # Fetch job description based on job_id
    job_description = fetch_job_description(job_id)

    if not job_description:
        return jsonify({"error": "Job not found"}), 404

    if not job_description:
        return jsonify({"error": "job_description is required"}), 400

    job_description = preprocess_text(job_description)
    
    candidates = load_candidates()
    if not candidates:
        return jsonify({"error": "No candidates found"}), 404
    
    ranked_candidates = rank_candidates(job_description, candidates)

    # Enforce correct field order in JSON response
    formatted_candidates = [
        {
            "similarity": candidate["similarity"],
            "id": candidate["id"],
            "name": candidate["name"],
            "description": candidate["description"],
            "skills": candidate["skills"],  # Single-line comma-separated skills
            "experiences": candidate["experiences"],
            "education": candidate["education"]
        }
        for candidate in ranked_candidates
    ]

    return jsonify({"ranked_candidates": formatted_candidates[:10]}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
