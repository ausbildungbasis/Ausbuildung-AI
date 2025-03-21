# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sshtunnel import SSHTunnelForwarder
from config import STAGING_CONFIG, LIVE_CONFIG  # Import configurations

# Download NLTK requirements
nltk.download('punkt')
nltk.download('punkt_tab')

# Use the German Stemmer
stemmer = SnowballStemmer("german")

# Connect to MySQL through SSH tunnel
def connect_db(ssh_host, ssh_username, ssh_password, ssh_port, db_host, db_user, db_password, db_name):
    """
    Connect to a MySQL database through an SSH tunnel.

    :param ssh_host: SSH server hostname
    :param ssh_username: SSH username
    :param ssh_password: SSH password
    :param ssh_port: SSH port
    :param db_host: Database hostname
    :param db_user: Database username
    :param db_password: Database password
    :param db_name: Database name
    :return: MySQL connection object
    """
    try:
        with SSHTunnelForwarder(
            (ssh_host, ssh_port),  # SSH server details
            ssh_username=ssh_username,
            ssh_password=ssh_password,
            remote_bind_address=(db_host, 3306)  # Database server details
        ) as tunnel:
            # Connect to the database through the SSH tunnel
            conn = mysql.connector.connect(
                host='127.0.0.1',  # Localhost because of the tunnel
                port=tunnel.local_bind_port,  # Local port forwarded by the tunnel
                user=db_user,
                password=db_password,
                database=db_name
            )
            return conn
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        return None

# Fetch and process candidate data
def load_candidates(server_type):
    """
    Load candidates from the specified server (Staging or Live).

    :param server_type: 'staging' or 'live'
    :return: List of processed candidates
    """
    # Choose the appropriate configuration
    if server_type == 'staging':
        config = STAGING_CONFIG
    elif server_type == 'live':
        config = LIVE_CONFIG
    else:
        raise ValueError("Invalid server type. Use 'staging' or 'live'.")

    # Connect to the database
    conn = connect_db(**config)
    if not conn:
        return []

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

@app.route('/', methods=['GET'])
def rank_candidates_api():
    job_description = 'Hard Working'
    server_type = request.args.get('server', 'staging')  # Default to staging

    if not job_description:
        return jsonify({"error": "job_description is required"}), 400

    job_description = preprocess_text(job_description)
    
    candidates = load_candidates(server_type)
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
