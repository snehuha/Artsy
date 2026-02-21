import streamlit as st
import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Artsy",
    layout="wide"
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    with open("../model/features.pkl", "rb") as f:
        features = pickle.load(f)

    with open("../model/image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)

    return features, image_paths

features, image_paths = load_data()

# -------------------- IMAGE UTILS --------------------
def get_dominant_color(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(img)

    return kmeans.cluster_centers_[0]

def color_to_mood(color):
    r, g, b = color

    if r < 80 and g < 80 and b < 80:
        return "dark"
    if b > r and b > g:
        return "calm"
    if r > 150 and g < 120:
        return "energetic"
    return "neutral"

# -------------------- RECOMMENDATION LOGIC --------------------
def recommend_art(index, top_n=25):
    sim = cosine_similarity([features[index]], features)[0]
    return sim.argsort()[::-1][1:top_n+1]

def recommend_with_mood(index, mood, top_n=5):
    base = recommend_art(index)
    final = []

    for i in base:
        color = get_dominant_color(image_paths[i])
        img_mood = color_to_mood(color)

        if img_mood == mood or img_mood == "neutral":
            final.append(i)

        if len(final) == top_n:
            break

    return final

# -------------------- MOOD-BASED GALLERY --------------------
def filter_gallery_by_mood(mood, max_items=30):
    gallery = []

    for i, path in enumerate(image_paths):
        color = get_dominant_color(path)
        img_mood = color_to_mood(color)

        if img_mood == mood or img_mood == "neutral":
            gallery.append((i, path))

        if len(gallery) == max_items:
            break

    return gallery


# -------------------- UI HEADER --------------------
st.markdown(
    "<h1 style='text-align:center;'> Artsy</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:18px;'>Art that feels like you</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------- MOOD INPUT --------------------
st.markdown("### How are you feeling today?")
mood = st.selectbox(
    "",
    ["calm", "energetic", "dark"]
)

st.divider()

# -------------------- GALLERY (MOOD-AWARE) --------------------
st.markdown("### Choose an artwork you like")

gallery_items = filter_gallery_by_mood(mood)

cols = st.columns(4)
selected_index = None

for i, (idx, path) in enumerate(gallery_items):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with cols[i % 4]:
        st.image(img, use_container_width=True)
        if st.button("Select", key=f"gallery_{i}"):
            selected_index = idx

# -------------------- RECOMMENDATIONS --------------------
if selected_index is not None:
    st.divider()
    st.markdown("### ✨ Recommended for you")

    recs = recommend_with_mood(selected_index, mood)

    rec_cols = st.columns(5)
    for i, idx in enumerate(recs):
        img = cv2.imread(image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with rec_cols[i]:
            st.image(img, use_container_width=True)
