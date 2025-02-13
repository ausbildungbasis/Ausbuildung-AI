import mysql.connector
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords (if not already installed)
nltk.download('punkt')
nltk.download('stopwords')

# Load German stopwords
STOPWORDS = set(stopwords.words('german'))

DB_CONFIG = {
    "host": "test.ausbildungsbasis.de",
    "user": "uaixkdmalwgpa",
    "password": "Ausbildungsbasis123?.",
    "database": "dbbjv8sgihuufp"
}

def connect_db():
    """Establish MySQL database connection."""
    return mysql.connector.connect(**DB_CONFIG)

def preprocess_text(text):
    """Tokenize, remove stopwords, and lowercase text for better similarity matching."""
    tokens = word_tokenize(text.lower())  # Tokenization & Lowercasing
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in STOPWORDS]  
    return " ".join(filtered_tokens)  # Return cleaned text

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
    conn.close()  # Close connection

    if not candidates:
        return []

    processed_candidates = []
    for candidate in candidates:
        full_text = f"Beschreibung: {candidate['description']}\nFähigkeiten: {candidate['skills']}\nErfahrung: {candidate['experiences']}\nAusbildung: {candidate['education']}"
        processed_candidates.append({
            "id": candidate["id"],
            "name": f"{candidate['firstname']} {candidate['name']}",
            "text": preprocess_text(full_text)  # Preprocess text
        })
    
    return processed_candidates  

def rank_candidates(job_description, candidates):
    """Rank candidates based on similarity to job description using TF-IDF + Cosine Similarity."""
    job_description = preprocess_text(job_description)  # Preprocess job description
    candidate_texts = [candidate["text"] for candidate in candidates]  

    # Compute TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_description] + candidate_texts)

    # Compute Cosine Similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    for i, candidate in enumerate(candidates):
        candidate["similarity"] = similarity_scores[i]

    return sorted(candidates, key=lambda x: x["similarity"], reverse=True)

def main():
    """Main function to rank candidates based on job description."""
    job_description = "hard working and manufacturing sales representative"

    candidates = load_candidates()  # Fetch candidates
    if not candidates:
        print("No candidates in database found.")
        return

    ranked_candidates = rank_candidates(job_description, candidates)

    print("\nTop-Kandidaten:\n")
    for i, candidate in enumerate(ranked_candidates[:10]):
        print(f"{i+1}. {candidate['name']} - Ähnlichkeit: {candidate['similarity']:.2f}")

if __name__ == "__main__":
    main()