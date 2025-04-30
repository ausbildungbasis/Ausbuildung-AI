from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# MySQL config
DB_CONFIG = {
    "host": "test.ausbildungsbasis.de",
    "user": "uaixkdmalwgpa",
    "password": "Ausbildungsbasis123?.",
    "database": "dbbjv8sgihuufp"
}

stemmer = SnowballStemmer("german")

def connect_db():
    return mysql.connector.connect(**DB_CONFIG)

# Preprocess text
def preprocess_text(text):
    words = word_tokenize(text.lower(), language="german")
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

# Recommend jobs to a candidate
@app.route('/recommend/<int:candidate_id>', methods=['GET'])
def recommend_jobs_for_candidate(candidate_id):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # Fetch candidate's full profile
    cursor.execute("""
        SELECT 
            CONCAT(
                t.description, ' ',
                IFNULL(GROUP_CONCAT(DISTINCT s.name SEPARATOR ' '), ''), ' ',
                IFNULL(GROUP_CONCAT(DISTINCT je.jobtitle SEPARATOR ' '), ''), ' ',
                IFNULL(GROUP_CONCAT(DISTINCT sc.degree SEPARATOR ' '), '')
            ) AS profile
        FROM trainees AS t
        LEFT JOIN trainee_skills AS ts ON ts.trainee_id = t.id
        LEFT JOIN skills AS s ON ts.skill_id = s.id  
        LEFT JOIN job_experiences AS je ON je.trainee_id = t.id
        LEFT JOIN school_careers AS sc ON sc.trainee_id = t.id 
        WHERE t.id = %s
        GROUP BY t.id;
    """, (candidate_id,))
    candidate = cursor.fetchone()

    if not candidate or not candidate["profile"]:
        conn.close()
        return jsonify({"error": "Candidate profile not found"}), 404

    candidate_text = preprocess_text(candidate["profile"])

    # Fetch all job listings
    cursor.execute("""
        SELECT id, jobtitle, description 
        FROM joblistings
    """)
    jobs = cursor.fetchall()
    conn.close()

    if not jobs:
        return jsonify({"error": "No job listings found"}), 404

    # Prepare texts for similarity
    job_texts = [preprocess_text(f"{job['jobtitle']} {job['description']}") for job in jobs]
    texts = [candidate_text] + job_texts

    # TF-IDF + similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

    for i, job in enumerate(jobs):
        job["similarity"] = f"{round(similarity_scores[i] * 100, 2)}%"

    # Sort jobs by similarity
    ranked_jobs = sorted(jobs, key=lambda j: float(j["similarity"].strip('%')), reverse=True)

    return jsonify({"recommended_jobs": ranked_jobs[:100]}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
