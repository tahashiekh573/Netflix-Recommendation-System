





# Netflix Content Recommendation System
# Task 1: Baseline vs Improved Recommendation Model

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Configuration

DATASET_PATH = "Dataset.csv"
NUMBER_OF_RECOMMENDATIONS = 5
SELECTED_TITLE = "Dick Johnson Is Dead"


# Step 1: Load Dataset

df = pd.read_csv(DATASET_PATH)

content_df = df[
    [
        "title",
        "type",
        "director",
        "country",
        "rating",
        "listed_in",
        "release_year"
    ]
].copy()

text_columns = [
    "title",
    "type",
    "director",
    "country",
    "rating",
    "listed_in"
]

for column in text_columns:
    content_df[column] = (
        content_df[column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

content_df["release_year"] = content_df["release_year"].fillna(
    content_df["release_year"].median()
)


# Step 2: Create TF-IDF Features

content_df["content_features"] = (
    content_df["type"] + " "
    + content_df["director"] + " "
    + content_df["country"] + " "
    + content_df["rating"] + " "
    + content_df["listed_in"]
)

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    content_df["content_features"]
)

similarity_matrix = cosine_similarity(tfidf_matrix)


# Step 3: Baseline Recommendation Model

def baseline_recommendations(title, number_of_recommendations=5):

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

    selected_indices = similar_indices[
        :number_of_recommendations
    ]

    recommendations = content_df.loc[
        selected_indices,
        [
            "title",
            "type",
            "country",
            "rating",
            "listed_in"
        ]
    ].copy()

    recommendations["score"] = [
        similarity_scores[index]
        for index in selected_indices
    ]

    recommendations.reset_index(drop=True, inplace=True)

    return recommendations


# Step 4: Genre Similarity

def calculate_genre_similarity(genres_a, genres_b):

    set_a = {
        genre.strip()
        for genre in genres_a.split(",")
        if genre.strip()
    }

    set_b = {
        genre.strip()
        for genre in genres_b.split(",")
        if genre.strip()
    }

    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    return intersection / union if union else 0.0


# Step 5: Improved Recommendation Model

def improved_recommendations(
    title,
    number_of_recommendations=5
):

    normalized_title = title.lower().strip()

    matches = content_df[
        content_df["title"] == normalized_title
    ]

    if matches.empty:
        return pd.DataFrame()

    title_index = matches.index[0]

    selected_type = content_df.loc[
        title_index,
        "type"
    ]

    selected_genres = content_df.loc[
        title_index,
        "listed_in"
    ]

    candidates = content_df[
        (content_df["type"] == selected_type)
        & (content_df.index != title_index)
    ]

    scores = []

    for index in candidates.index:

        content_score = similarity_matrix[
            title_index,
            index
        ]

        genre_score = calculate_genre_similarity(
            selected_genres,
            content_df.loc[index, "listed_in"]
        )

        final_score = (
            0.65 * content_score
            + 0.35 * genre_score
        )

        scores.append(
            (
                index,
                content_score,
                genre_score,
                final_score
            )
        )

    scores.sort(
        key=lambda item: item[3],
        reverse=True
    )

    selected_scores = scores[
        :number_of_recommendations
    ]

    selected_indices = [
        item[0]
        for item in selected_scores
    ]

    recommendations = content_df.loc[
        selected_indices,
        [
            "title",
            "type",
            "country",
            "rating",
            "listed_in"
        ]
    ].copy()

    recommendations["score"] = [
        item[3]
        for item in selected_scores
    ]

    recommendations.reset_index(drop=True, inplace=True)

    return recommendations


# Step 6: Evaluation

def calculate_average_score(recommendations):

    if recommendations.empty:
        return 0.0

    return recommendations["score"].mean()


# Step 7: Run Both Models

baseline = baseline_recommendations(
    SELECTED_TITLE,
    NUMBER_OF_RECOMMENDATIONS
)

improved = improved_recommendations(
    SELECTED_TITLE,
    NUMBER_OF_RECOMMENDATIONS
)

baseline_score = calculate_average_score(
    baseline
)

improved_score = calculate_average_score(
    improved
)

score_difference = improved_score - baseline_score


# Step 8: Display Results

print("=" * 75)
print("NETFLIX RECOMMENDATION SYSTEM")
print("BASELINE VS IMPROVED MODEL")
print("=" * 75)

print(f"\nDataset Records : {len(df)}")
print(f"TF-IDF Features : {tfidf_matrix.shape[1]}")
print(f"Selected Title  : {SELECTED_TITLE}")


print("\nBASELINE MODEL")
print("-" * 75)

for index, row in baseline.iterrows():

    print(
        f"{index + 1}. "
        f"{row['title'].title()} "
        f"({row['type'].title()}) "
        f"- Score: {row['score']:.4f}"
    )

print(
    f"\nBaseline Average Score: "
    f"{baseline_score:.4f}"
)


print("\nIMPROVED MODEL")
print("-" * 75)

for index, row in improved.iterrows():

    print(
        f"{index + 1}. "
        f"{row['title'].title()} "
        f"({row['type'].title()}) "
        f"- Score: {row['score']:.4f}"
    )

print(
    f"\nImproved Average Score: "
    f"{improved_score:.4f}"
)


print("\nMODEL COMPARISON")
print("-" * 75)

print(
    f"Baseline Average Score : "
    f"{baseline_score:.4f}"
)

print(
    f"Improved Average Score : "
    f"{improved_score:.4f}"
)

print(
    f"Score Difference        : "
    f"{score_difference:.4f}"
)


if score_difference > 0:
    print("\n✓ Improved model produced a higher recommendation score.")

elif score_difference < 0:
    print("\n✓ Baseline model produced a higher recommendation score.")

else:
    print("\n✓ Both models produced the same average score.")


print("\nPROJECT STATUS")
print("-" * 75)

print("✓ Dataset loaded")
print("✓ TF-IDF features generated")
print("✓ Baseline model evaluated")
print("✓ Genre similarity added")
print("✓ Improved model evaluated")
print("✓ Models compared")

print("\nBASELINE VS IMPROVED COMPARISON COMPLETED")