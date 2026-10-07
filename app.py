import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Configuration

DATASET_PATH = "Dataset.csv"
DEFAULT_TITLE = "dick johnson is dead"
DEFAULT_RECOMMENDATIONS = 5


# Page Configuration

st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom Styling

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .recommendation-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.35);
        margin-bottom: 15px;
        min-height: 250px;
    }

    .recommendation-card h3 {
        margin-bottom: 18px;
    }

    .recommendation-card p {
        margin: 8px 0;
    }

    .score {
        font-size: 18px;
        font-weight: 700;
        margin-top: 15px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.35);
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 15px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Dataset Loading

@st.cache_data
def load_dataset():

    data = pd.read_csv(DATASET_PATH)

    required_columns = [
        "title",
        "type",
        "director",
        "country",
        "rating",
        "listed_in"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing dataset columns: "
            + ", ".join(missing_columns)
        )

    for column in required_columns:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    return data


# Model Building

@st.cache_resource
def build_model(data):

    content_features = (
        data["type"] + " "
        + data["director"] + " "
        + data["country"] + " "
        + data["rating"] + " "
        + data["listed_in"] + " "
        + data["listed_in"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        content_features
    )

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    return (
        vectorizer,
        tfidf_matrix,
        similarity_matrix
    )


# Genre Similarity

def calculate_genre_similarity(
    genres_a,
    genres_b
):

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

    intersection = len(
        set_a.intersection(set_b)
    )

    union = len(
        set_a.union(set_b)
    )

    return (
        intersection / union
        if union
        else 0.0
    )


# Baseline Recommendation Model

def get_baseline_recommendations(
    title,
    number_of_recommendations
):

    matches = df[
        df["title"] == title
    ]

    if matches.empty:
        return pd.DataFrame()

    title_index = matches.index[0]

    similarity_scores = similarity_matrix[
        title_index
    ]

    similar_indices = similarity_scores.argsort()[
        ::-1
    ]

    similar_indices = [
        index
        for index in similar_indices
        if index != title_index
    ]

    selected_indices = similar_indices[
        :number_of_recommendations
    ]

    recommendations = []

    for index in selected_indices:

        recommendations.append(
            {
                "title": df.loc[
                    index,
                    "title"
                ].title(),

                "type": df.loc[
                    index,
                    "type"
                ].title(),

                "score": similarity_scores[index]
            }
        )

    return pd.DataFrame(
        recommendations
    )


# Improved Recommendation Model

def get_improved_recommendations(
    title,
    number_of_recommendations
):

    matches = df[
        df["title"] == title
    ]

    if matches.empty:
        return pd.DataFrame()

    title_index = matches.index[0]

    selected_type = df.loc[
        title_index,
        "type"
    ]

    selected_genres = df.loc[
        title_index,
        "listed_in"
    ]

    candidates = df[
        (df["type"] == selected_type)
        & (df.index != title_index)
    ]

    scores = []

    for index in candidates.index:

        content_score = similarity_matrix[
            title_index,
            index
        ]

        genre_score = calculate_genre_similarity(
            selected_genres,
            df.loc[
                index,
                "listed_in"
            ]
        )

        final_score = (
            0.65 * content_score
            + 0.35 * genre_score
        )

        scores.append(
            {
                "index": index,
                "content_score": content_score,
                "genre_score": genre_score,
                "final_score": final_score
            }
        )

    scores.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    selected_scores = scores[
        :number_of_recommendations
    ]

    recommendations = []

    for item in selected_scores:

        index = item["index"]

        recommendations.append(
            {
                "title": df.loc[
                    index,
                    "title"
                ].title(),

                "type": df.loc[
                    index,
                    "type"
                ].title(),

                "country": df.loc[
                    index,
                    "country"
                ].title(),

                "rating": df.loc[
                    index,
                    "rating"
                ].upper(),

                "genres": df.loc[
                    index,
                    "listed_in"
                ].title(),

                "content_score": item[
                    "content_score"
                ],

                "genre_score": item[
                    "genre_score"
                ],

                "score": item[
                    "final_score"
                ]
            }
        )

    return pd.DataFrame(
        recommendations
    )


# Load Data and Model

try:

    df = load_dataset()

    (
        vectorizer,
        tfidf_matrix,
        similarity_matrix
    ) = build_model(df)

except Exception as error:

    st.error(
        f"Application error: {error}"
    )

    st.stop()


# Header

st.markdown(
    """
    <div class="main-title">
        🎬 Netflix Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-powered content recommendations using
        TF-IDF, cosine similarity and genre analysis
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar

with st.sidebar:

    st.header("⚙️ Recommendation Settings")

    number_of_recommendations = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=DEFAULT_RECOMMENDATIONS
    )

    st.divider()

    st.subheader("📚 Dataset")

    st.metric(
        "Netflix Titles",
        f"{len(df):,}"
    )

    st.metric(
        "TF-IDF Features",
        f"{tfidf_matrix.shape[1]:,}"
    )

    st.divider()

    st.caption(
        "Machine Learning Internship — Task 1"
    )


# Title Selection

st.markdown(
    """
    <div class="section-title">
        🔎 Find Similar Netflix Titles
    </div>
    """,
    unsafe_allow_html=True
)

title_options = sorted(
    df["title"].dropna().unique()
)

default_index = (
    title_options.index(DEFAULT_TITLE)
    if DEFAULT_TITLE in title_options
    else 0
)

selected_title = st.selectbox(
    "Select a Netflix title",
    title_options,
    index=default_index
)


# Recommendation Button

generate = st.button(
    "🎯 Get Recommendations",
    use_container_width=True
)


if generate:

    recommendations = get_improved_recommendations(
        selected_title,
        number_of_recommendations
    )

    baseline = get_baseline_recommendations(
        selected_title,
        number_of_recommendations
    )

    if recommendations.empty:

        st.warning(
            "No recommendations found for the selected title."
        )

        st.stop()

    if baseline.empty:

        st.warning(
            "Baseline recommendations could not be generated."
        )

        st.stop()


    # Model Scores

    baseline_average = baseline[
        "score"
    ].mean()

    improved_average = recommendations[
        "score"
    ].mean()

    score_difference = (
        improved_average
        - baseline_average
    )

    improvement_percentage = (
        score_difference
        / baseline_average
        * 100
        if baseline_average > 0
        else 0
    )


    # Recommendation Results

    st.divider()

    st.markdown(
        f"""
        <div class="section-title">
            🎬 Recommendations for:
            {selected_title.title()}
        </div>
        """,
        unsafe_allow_html=True
    )


    columns = st.columns(2)

    for index, row in recommendations.iterrows():

        with columns[index % 2]:

            st.markdown(
                f"""
                <div class="recommendation-card">

                <h3>
                {index + 1}. {row['title']}
                </h3>

                <p>
                <b>Type:</b>
                {row['type']}
                </p>

                <p>
                <b>Genre:</b>
                {row['genres']}
                </p>

                <p>
                <b>Rating:</b>
                {row['rating']}
                </p>

                <p>
                <b>Country:</b>
                {row['country']}
                </p>

                <p class="score">
                ⭐ Similarity Score:
                {row['score']:.2%}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


    # Model Performance

    st.divider()

    st.markdown(
        """
        <div class="section-title">
            📊 Model Performance
        </div>
        """,
        unsafe_allow_html=True
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Baseline Score",
            f"{baseline_average:.2%}"
        )

    with metric2:

        st.metric(
            "Improved Score",
            f"{improved_average:.2%}"
        )

    with metric3:

        st.metric(
            "Score Difference",
            f"{score_difference:.2%}"
        )


    # Model Comparison Chart

    chart_data = pd.DataFrame(
        {
            "Model": [
                "Baseline",
                "Improved"
            ],

            "Average Score": [
                baseline_average,
                improved_average
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Model")
    )


    # Project Statistics

    st.divider()

    st.markdown(
        """
        <div class="section-title">
            📈 Project Statistics
        </div>
        """,
        unsafe_allow_html=True
    )

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:

        st.metric(
            "Dataset Records",
            f"{len(df):,}"
        )

    with stat2:

        st.metric(
            "TF-IDF Features",
            f"{tfidf_matrix.shape[1]:,}"
        )

    with stat3:

        st.metric(
            "Recommendations",
            number_of_recommendations
        )

    with stat4:

        st.metric(
            "Score Improvement",
            f"{improvement_percentage:.1f}%"
        )


    # Methodology

    st.divider()

    st.markdown(
        """
        <div class="section-title">
            🧠 ML Methodology
        </div>
        """,
        unsafe_allow_html=True
    )

    method1, method2, method3 = st.columns(3)

    with method1:

        st.markdown(
            """
            ### 1. Feature Engineering

            Netflix metadata including type,
            director, country, rating and genres
            is combined into content features.
            """
        )

    with method2:

        st.markdown(
            """
            ### 2. TF-IDF & Similarity

            Text features are converted into
            numerical vectors using TF-IDF and
            compared using cosine similarity.
            """
        )

    with method3:

        st.markdown(
            """
            ### 3. Improved Ranking

            Genre similarity and content similarity
            are combined using a weighted scoring
            approach to rank recommendations.
            """
        )


    # Evaluation Note

    st.info(
        "Evaluation scores represent custom content-similarity "
        "metrics. They should not be interpreted as prediction "
        "accuracy because the dataset does not contain "
        "ground-truth user relevance labels."
    )


# Footer

st.divider()

st.markdown(
    """
    <div class="footer">
        Netflix Content Recommendation System
        |
        Machine Learning Internship — Task 1
        |
        Content-Based Recommendation
    </div>
    """,
    unsafe_allow_html=True
)