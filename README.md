# Resume Ranking API

## Overview
This project is a **Flask-based API** that ranks candidate resumes based on a given job description using **TF-IDF vectorization and cosine similarity**. The system retrieves candidate data from a **MySQL database**, processes it using **NLTK for text preprocessing**, and returns the top-ranked candidates based on similarity scores.

## Features
- Fetches candidate data (skills, experiences, and education) from a **MySQL database**.
- Preprocesses text using **tokenization and stemming** (German language support).
- Uses **TF-IDF vectorization** to convert text into numerical features.
- Computes **cosine similarity** to rank candidates based on a given job description.
- Provides an API endpoint to return the **top 10 ranked candidates**.
- Supports **CORS** for cross-origin requests.

## Technologies Used
- **Flask** (for building the API)
- **Flask-CORS** (for enabling cross-origin requests)
- **MySQL Connector** (for database operations)
- **NLTK** (for text processing - tokenization & stemming)
- **Scikit-learn** (for TF-IDF vectorization & similarity calculation)

## Installation & Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/ausbildungbasis/Ausbuildung-AI.git
cd resume-ranker
``` 

### **2. Install Dependencies**
Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate   # For MacOS/Linux
venv\Scripts\activate     # For Windows
```
Then install the required packages:
```bash
pip install -r requirements.txt
```

### **3. Set Up MySQL Database**
Ensure you have a MySQL database running with the correct credentials (update in `DB_CONFIG` inside the script):
```python
DB_CONFIG = {
    "host": "",
    "user": "",
    "password": "",
    "database": ""
}
```

### **4. Run the API**
```bash
python app.py
```
The API will start on `http://0.0.0.0:5000/`.

## API Usage
### **GET /**
Fetches the **top 10 ranked candidates** for a job description.
#### **Response Format:**
```json
{
    "ranked_candidates": [
        {
            "similarity": in Percentage,
            "id": ,
            "name": "",
            "description": "",
            "skills": "",
            "experiences": "",
            "education": ""
        },
        ...
    ]
}
```
## License
This project is private and licensed under the **MIT License**.

## Author
[Ausbuildungsbasis]  


