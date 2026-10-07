# 🎬 Netflix Content Recommendation System

A professional content-based Netflix recommendation system developed using Python, Scikit-Learn, Pandas, and Streamlit.

The system recommends similar Netflix titles by analyzing content metadata such as genres, content type, director, country, rating, and other available attributes.

---

## 📌 Project Overview

This project was developed as part of the Machine Learning Internship — Task 1.

The objective is to build a content-based recommendation system capable of identifying Netflix titles with similar characteristics and generating personalized recommendations for a selected title.

The project includes both:

- Baseline Recommendation Model
- Improved Recommendation Model

The improved model combines TF-IDF content similarity with genre similarity to produce more relevant recommendations.

---

## 🎯 Objectives

The main objectives of this project are:

- Prepare Netflix content-related features.
- Convert text information into machine-readable numerical features.
- Apply TF-IDF vectorization.
- Calculate cosine similarity between Netflix titles.
- Generate recommendations for selected titles.
- Improve recommendation ranking using genre similarity.
- Compare baseline and improved recommendation performance.
- Provide an interactive Streamlit web application.

---

## 🧠 Machine Learning Approach

The recommendation system follows a content-based filtering approach.

### 1. Feature Engineering

The system uses available Netflix metadata including:

- Content Type
- Director
- Country
- Rating
- Genres / Categories

These attributes are combined to create content features for each title.

### 2. TF-IDF Vectorization

Text-based content features are converted into numerical vectors using:

```text
TF-IDF

TF-IDF helps represent the importance of words and categories within the Netflix dataset.

3. Cosine Similarity

Cosine similarity is used to calculate how similar two Netflix titles are based on their TF-IDF representations.

The resulting similarity score is used by the baseline recommendation model.

4. Genre Similarity

The improved model additionally analyzes genre overlap between titles.

Genre similarity is combined with content similarity to improve the recommendation ranking.

5. Final Recommendation Score

The improved model uses a weighted scoring approach:

Final Score =
0.65 × Content Similarity
+
0.35 × Genre Similarity

Only titles with the same content type are prioritized by the improved model.

📊 Dataset

The project uses a Netflix titles dataset containing:

8,790 Records

The main dataset columns include:

Column	Description
show_id	Unique Netflix content identifier
type	Movie or TV Show
title	Netflix title
director	Director information
country	Production country
date_added	Date added to Netflix
release_year	Original release year
rating	Content rating
duration	Movie/TV duration
listed_in	Genres and categories
🚀 Application Features

The Streamlit application provides:

🔎 Title Selection

Users can select any available Netflix title from the dataset.

🎯 Recommendation Control

Users can choose how many recommendations they want using the recommendation slider.

🎬 Recommendation Cards

Each recommendation displays:

Title
Content Type
Genre
Rating
Country
Similarity Score
📊 Model Performance

The application compares:

Baseline Average Score
Improved Average Score
Score Difference
📈 Performance Visualization

A chart visually compares the average similarity scores of the baseline and improved models.

📚 Project Statistics

The application displays:

Dataset Records
TF-IDF Features
Number of Recommendations
Score Improvement
🧠 ML Methodology

The application also explains the main machine learning workflow used by the recommendation system.

📈 Example Results

For the selected title:

Dick Johnson Is Dead

the improved model generated recommendations including:

Nova: Killer Floods
She Did That
Crip Camp: A Disability Revolution
American Factory: A Conversation With The Obamas
Murder To Mercy: The Cyntoia Brown Story

Example similarity scores:

Nova: Killer Floods                         68.20%
She Did That                               68.20%
Crip Camp: A Disability Revolution         66.37%
American Factory: A Conversation With
The Obamas                                 65.97%
Murder To Mercy: The Cyntoia Brown Story   64.96%

The application also provides a baseline-versus-improved model comparison.

📊 Model Evaluation

The project compares the baseline and improved recommendation approaches.

Example application output:

Baseline Average Score : 47.28%
Improved Average Score : 65.73%
Score Difference       : 18.45%

The improved model produces a higher average content-similarity score for the selected recommendation set.

Note: These scores are similarity-based evaluation metrics. They should not be interpreted as prediction accuracy because the dataset does not contain ground-truth user relevance labels.

🛠️ Technologies Used
Programming Language
Python 3.11
Machine Learning
Scikit-Learn
TF-IDF Vectorization
Cosine Similarity
Content-Based Filtering
Data Processing
Pandas
NumPy
Web Application
Streamlit
Development Environment
Visual Studio Code
PowerShell
Git / GitHub
📁 Project Structure
Netflix-Recommendation-System/
│
├── Dataset.csv
│
├── task1_recommendation.py
│
├── app.py
│
├── README.md
│
└── screenshots/
    ├── home.png
    ├── recommendations.png
    └── model-performance.png
⚙️ Installation

Clone the project:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate to the project directory:

cd Netflix-Recommendation-System

Install required libraries:

python -m pip install pandas numpy scikit-learn streamlit
▶️ Run the Machine Learning Model

Run the command:

python task1_recommendation.py

The program will:

Load the Netflix dataset.
Prepare content features.
Generate TF-IDF vectors.
Calculate similarity.
Generate recommendations.
Compare recommendation models.
Display evaluation results.
🌐 Run the Streamlit Application

Start the web application using:

python -m streamlit run app.py

The application will be available locally at:

http://localhost:8501

The port may change automatically if another Streamlit instance is already running.

🧪 Testing

The application was tested using multiple Netflix titles and recommendation counts.

Testing verifies:

Dataset loading
Feature extraction
TF-IDF generation
Similarity calculation
Genre similarity
Recommendation generation
Model comparison
Streamlit interface
User title selection
Recommendation count control
💡 Key Learning Outcomes

This project demonstrates practical understanding of:

Recommendation Systems
Content-Based Filtering
Natural Language Processing fundamentals
Feature Engineering
TF-IDF
Cosine Similarity
Similarity-Based Ranking
Scikit-Learn
Data Processing with Pandas
Streamlit Application Development
Model Evaluation
Machine Learning Project Presentation
🔮 Future Improvements

Possible future improvements include:

User-based personalized recommendations
Collaborative filtering
Hybrid recommendation systems
User rating integration
Search functionality
Netflix-style movie posters
Movie descriptions
Advanced recommendation explanations
Recommendation history
User preference profiles
Model optimization for larger datasets
Cloud deployment
⚠️ Evaluation Limitation

This project uses a content-based similarity approach.

The provided dataset does not contain user interaction data such as:

User ratings
Watch history
Likes
Dislikes
Click-through data

Therefore, traditional recommendation metrics such as Precision@K, Recall@K, or NDCG cannot be directly calculated from the provided dataset without additional ground-truth interaction data.

The current evaluation focuses on similarity-based ranking and comparison between the baseline and improved approaches.

👨‍💻 Author

Muhammad Taha

Machine Learning Internship — Task 1

Netflix Content Recommendation System

📌 Project Status
✅ Dataset Preparation
✅ Feature Engineering
✅ TF-IDF Vectorization
✅ Cosine Similarity
✅ Baseline Recommendation Model
✅ Improved Recommendation Model
✅ Genre Similarity
✅ Model Comparison
✅ Streamlit Web Application
✅ Recommendation Evaluation
✅ Professional Documentation
⭐ Conclusion

The Netflix Content Recommendation System successfully demonstrates how machine learning and natural language processing techniques can be used to build a content-based recommendation engine.

The project progresses from data preparation and feature extraction to similarity calculation, recommendation generation, model improvement, evaluation, and deployment through an interactive Streamlit interface.