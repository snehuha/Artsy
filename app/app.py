import streamlit as st
import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os
import random


# ================= PATH FIX =================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Artsy",
    page_icon="🎨",
    layout="wide"
)


# ================= MODERN UI =================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0f172a, #020617);
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.title {
    text-align: center;
    font-size: 64px;
    font-weight: 800;
    background: linear-gradient(90deg,#a78bfa,#f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align:center;
    font-size:22px;
    color:#94a3b8;
    margin-bottom:40px;
}
.stButton>button {
    border-radius:12px;
    border:none;
    padding:10px;
    background:linear-gradient(90deg,#8b5cf6,#ec4899);
    color:white;
    font-weight:bold;
}
.stButton>button:hover {
    transform:scale(1.05);
}
section[data-testid="stSidebar"] {
    background: #020617;
}
</style>
""", unsafe_allow_html=True)


# ================= LOAD DATA =================

@st.cache_data
def load_data():
    feature_path = os.path.join(PROJECT_ROOT, "model", "features.pkl")
    image_path_file = os.path.join(PROJECT_ROOT, "model", "image_paths.pkl")

    with open(feature_path, "rb") as f:
        features = pickle.load(f)

    with open(image_path_file, "rb") as f:
        image_paths = pickle.load(f)

    return np.array(features), image_paths


features, image_paths = load_data()


# ================= IMAGE UTILS =================

def get_full_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)


# Pre-compute mood for every image once and cache it
@st.cache_data
def compute_all_moods():
    moods = {}
    for i, path in enumerate(image_paths):
        try:
            full_path = get_full_path(path)
            img = cv2.imread(full_path)
            if img is None:
                moods[i] = "neutral"
                continue
            img = cv2.resize(img, (50, 50))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = img.reshape((-1, 3)).astype(float)
            # Mean color is much faster than KMeans, same accuracy for mood
            r, g, b = pixels[:, 0].mean(), pixels[:, 1].mean(), pixels[:, 2].mean()

            if r < 70 and g < 70 and b < 70:
                moods[i] = "dark"
            elif b > r and b > g:
                moods[i] = "calm"
            elif r > 150:
                moods[i] = "energetic"
            else:
                moods[i] = "neutral"
        except:
            moods[i] = "neutral"
    return moods


# ================= RECOMMENDATION =================

def recommend_with_mood(index, mood, all_moods, top_n=5):
    similarity = cosine_similarity([features[index]], features)[0]
    similarity[index] = -1  # exclude self

    sorted_indices = similarity.argsort()[::-1]

    mood_match = []
    fallback = []

    for i in sorted_indices:
        img_mood = all_moods.get(i, "neutral")
        if img_mood == mood or img_mood == "neutral":
            mood_match.append(i)
        else:
            fallback.append(i)

        if len(mood_match) >= top_n:
            break

    result = mood_match[:top_n]

    # Pad with closest matches if not enough mood-matching results
    if len(result) < top_n:
        for i in fallback:
            if i not in result:
                result.append(i)
            if len(result) == top_n:
                break

    return result


# ================= GALLERY FILTER =================

def filter_gallery_by_mood(mood, all_moods, max_items=30):
    matching = [
        (i, image_paths[i])
        for i, m in all_moods.items()
        if m == mood or m == "neutral"
    ]
    # Shuffle so gallery varies each time mood is selected
    random.shuffle(matching)
    return matching[:max_items]


# ================= HEADER =================

st.markdown('<div class="title">Artsy</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover art that matches your soul</div>', unsafe_allow_html=True)


# ================= SIDEBAR =================

st.sidebar.title("🎭 Mood")

mood = st.sidebar.radio(
    "Choose your feeling",
    ["calm", "energetic", "dark"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Sneha")


# ================= SESSION STATE =================

if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

if "current_mood" not in st.session_state:
    st.session_state.current_mood = mood

# Reset selection when mood changes
if st.session_state.current_mood != mood:
    st.session_state.current_mood = mood
    st.session_state.selected_index = None


# ================= PRE-COMPUTE MOODS =================

with st.spinner("Loading artwork moods..."):
    all_moods = compute_all_moods()


# ================= GALLERY =================

st.markdown("### 🖼️ Choose an artwork")

gallery_items = filter_gallery_by_mood(mood, all_moods)

if not gallery_items:
    st.warning("No artworks found for this mood. Try another!")
else:
    cols = st.columns(4)
    for i, (idx, path) in enumerate(gallery_items):
        img = cv2.imread(get_full_path(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with cols[i % 4]:
            st.image(img, use_container_width=True)
            if st.button("Select", key=f"gallery_{i}"):
                st.session_state.selected_index = idx
                st.rerun()


# ================= RECOMMENDATIONS =================

if st.session_state.selected_index is not None:
    st.markdown("---")
    st.markdown("## ✨ Recommended for you")

    recs = recommend_with_mood(
        st.session_state.selected_index,
        mood,
        all_moods
    )

    if recs:
        rec_cols = st.columns(len(recs))
        for i, idx in enumerate(recs):
            img = cv2.imread(get_full_path(image_paths[idx]))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with rec_cols[i]:
                st.image(img, use_container_width=True)
                score = cosine_similarity(
                    [features[st.session_state.selected_index]],
                    [features[idx]]
                )[0][0]
                st.caption(f"Match: {score:.2f}")
    else:
        st.info("No recommendations found. Try selecting a different artwork.")