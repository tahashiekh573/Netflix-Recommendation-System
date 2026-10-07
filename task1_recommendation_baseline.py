# Netflix Content Recommendation System
# Task 1: Content-Based Recommendation System

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Configuration

DATASET_PATH = "Dataset.csv"
NUMBER_OF_RECOMMENDATIONS = 5


# Step 1: Prepare Content Features

df = pd.read_csv(DATASET_PATH)

content_columns = [
    "title",
    "type",
    "director",
    "country",
    "rating",
    "listed_in",
    "release_year"
]

content_df = df[content_columns].copy()

text_columns = [
    "title",
    "type",
    "director",
    "country",
    "rating",
    "listed_in"
]

for column in text_columns:
    content_df[column] = content_df[column].fillna("")

content_df["release_year"] = content_df["release_year"].fillna(
    content_df["release_year"].median()
)

for column in text_columns:
    content_df[column] = (
        content_df[column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

content_df["content_features"] = (
    content_df["type"] + " "
    + content_df["director"] + " "
    + content_df["country"] + " "
    + content_df["rating"] + " "
    + content_df["listed_in"]
)


# Step 2: Convert Text into TF-IDF Vectors

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    content_df["content_features"]
)


# Step 3: Calculate Content Similarity

similarity_matrix = cosine_similarity(tfidf_matrix)


# Step 4: Generate Recommendations

def recommend_titles(title, number_of_recommendations=5):
    """Return the most similar Netflix titles."""

    normalized_title = title.lower().strip()

    matches = content_df[
        content_df["title"] == normalized_title
    ]

    if matches.empty:
        return pd.DataFrame()

    title_index = matches.index[0]

    similarity_scores = similarity_matrix[title_index]

    similar_indices = similarity_scores.argsort()[::-1]

    similar_indices = [
        index
        for index in similar_indices
        if index != title_index
    ]

    recommended_indices = similar_indices[
        :number_of_recommendations
    ]

    recommendations = content_df.iloc[
        recommended_indices
    ][
        [
            "title",
            "type",
            "country",
            "rating",
            "listed_in"
        ]
    ].copy()

    recommendations["similarity_score"] = [
        similarity_scores[index]
        for index in recommended_indices
    ]

    recommendations.reset_index(drop=True, inplace=True)

    return recommendations


# Step 5: Evaluate Recommendation Quality

def evaluate_recommendations(recommendations):
    """Calculate basic recommendation quality metrics."""

    if recommendations.empty:
        return {
            "number_of_recommendations": 0,
            "average_similarity": 0.0,
            "highest_similarity": 0.0,
            "lowest_similarity": 0.0
        }

    scores = recommendations["similarity_score"]

    return {
        "number_of_recommendations": len(recommendations),
        "average_similarity": scores.mean(),
        "highest_similarity": scores.max(),
        "lowest_similarity": scores.min()
    }


# Generate Recommendations for Selected Title

selected_title = "Dick Johnson Is Dead"

recommendations = recommend_titles(
    selected_title,
    NUMBER_OF_RECOMMENDATIONS
)


# Evaluate Recommendations

evaluation = evaluate_recommendations(
    recommendations
)


# Display Final Results

print("=" * 65)
print("NETFLIX CONTENT RECOMMENDATION SYSTEM")
print("=" * 65)

print(f"\nDataset Records       : {len(df)}")
print(f"TF-IDF Features      : {tfidf_matrix.shape[1]}")
print(f"Selected Title       : {selected_title}")

print("\nRecommended Titles")
print("-" * 65)

if recommendations.empty:
    print("Title not found in the dataset.")

else:
    for index, row in recommendations.iterrows():
        print(
            f"{index + 1}. "
            f"{row['title'].title()} "
            f"({row['type'].title()}) "
            f"- Similarity: {row['similarity_score']:.4f}"
        )

print("\nRecommendation Evaluation")
print("-" * 65)

print(
    f"Recommendations Generated : "
    f"{evaluation['number_of_recommendations']}"
)

print(
    f"Average Similarity        : "
    f"{evaluation['average_similarity']:.4f}"
)

print(
    f"Highest Similarity        : "
    f"{evaluation['highest_similarity']:.4f}"
)

print(
    f"Lowest Similarity         : "
    f"{evaluation['lowest_similarity']:.4f}"
)

print("\nTask Status")
print("-" * 65)
print("✓ Content features prepared")
print("✓ TF-IDF vectorization completed")
print("✓ Cosine similarity calculated")
print("✓ Recommendations generated")
print("✓ Recommendation quality evaluated")

print("\nTASK 1 COMPLETED SUCCESSFULLY")

