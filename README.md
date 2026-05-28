Artsy
Mood-Aware Art Recommendation System using Deep Learning

Artsy is a content-based art recommendation system that recommends visually and aesthetically similar artworks using deep learning, computer vision, and mood-aware filtering. The system uses a pre-trained VGG16 Convolutional Neural Network to extract visual embeddings from artworks and applies cosine similarity to generate recommendations.

In addition to visual similarity, Artsy introduces a lightweight mood-aware recommendation layer using dominant color analysis and heuristic-based aesthetic filtering to create a more immersive and personalized art exploration experience.

📌 Features
🎨 Deep learning–based artwork recommendation
🧠 Feature extraction using pre-trained VGG16
🔍 Content-based similarity search using cosine similarity
🌈 Mood-aware gallery filtering
🎭 Dynamic UI styling based on selected mood
⚡ Fast recommendations using precomputed embeddings
🖼 Interactive Streamlit-based interface
📚 Works across multiple art styles and visual domains
🧠 How the System Works

The recommendation pipeline follows a content-based filtering architecture.

Artwork Image
      ↓
VGG16 Feature Extractor
      ↓
Feature Vector Embedding
      ↓
Cosine Similarity Search
      ↓
Ranked Recommendations
      ↓
Mood-Based Aesthetic Filtering
🔬 Technical Overview
1. Feature Extraction using VGG16

A pre-trained VGG16 CNN model is used as a fixed feature extractor.

VGG16(weights="imagenet", include_top=False)
Why VGG16?
Strong visual feature extraction capability
Pretrained on ImageNet (~14 million images)
Effective for transfer learning
Captures textures, shapes, and artistic patterns

The final classification layers are removed using:

include_top=False

This allows the model to output deep visual embeddings instead of object labels.

2. Image Embeddings

Each artwork image is:

resized to 224 × 224
preprocessed using ImageNet normalization
passed through VGG16

The resulting activation tensor is flattened into a feature vector.

Artwork → Feature Vector (Numerical Representation)

These vectors represent:

composition
texture
shape
visual structure
color patterns
3. Recommendation Engine

The system uses cosine similarity to compare artwork embeddings.

Formula
cos(θ)=(A⋅B)/(∣∣A∣∣∣∣B∣∣)

The recommendation pipeline:

Extracts the query artwork vector
Computes similarity with all artwork vectors
Sorts similarity scores
Returns the top-K most similar artworks
4. Mood-Based Filtering

ArtWhisper introduces a lightweight mood-aware personalization layer.

Workflow
Extract dominant artwork color using K-Means clustering
Map dominant color to aesthetic mood categories
Filter recommendations based on selected mood
Supported moods
Calm
Energetic
Dark

This layer enhances personalization without claiming psychological or emotion inference.

🎨 UI/UX Design

The interface is built using Streamlit with:

responsive artwork gallery
mood-aware UI styling
dynamic background colors
interactive artwork selection

The design philosophy emphasizes:

minimalism
emotional immersion
intuitive exploration
🛠 Tech Stack
Machine Learning
TensorFlow / Keras
VGG16
Scikit-learn
Computer Vision
OpenCV
NumPy
Frontend / UI
Streamlit
Custom CSS
Data Handling
Pickle
Pandas
📂 Project Structure
ARTSY/
│
├── app/
│   └── app.py
│
├── data/
│   └── images/
│
├── model/
│   ├── features.pkl
│   └── image_paths.pkl
│
├── notebooks/
│   ├── 1.load_images.ipynb
│   └── 2.extract_features.ipynb
│
└── README.md
⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/yourusername/artwhisper.git
cd artwhisper
2. Install Dependencies
pip install tensorflow streamlit opencv-python scikit-learn matplotlib numpy pandas
3. Run Feature Extraction Notebook

Open:

notebooks/2.extract_features.ipynb

This generates:

features.pkl
image_paths.pkl

inside the model/ directory.

4. Launch the Streamlit App
cd app
streamlit run app.py
📊 Dataset

The project uses an artwork image dataset containing:

paintings
drawings
engravings
iconography

Dataset structure:

training_set/
├── drawings/
├── engraving/
├── iconography/
└── painting/

🚀 Future Improvements
🎭 Text-based mood input
🧠 NLP-powered sentiment analysis
❤️ Save user favorites
☁️ Cloud deployment
🎨 Artist/style-specific recommendations
🔎 Semantic search using CLIP embeddings
📱 Mobile-responsive UI
🧪 Key ML Concepts Used
Transfer Learning
CNN Feature Extraction
Image Embeddings
Cosine Similarity
Content-Based Recommendation Systems
K-Means Clustering
Mood-Based Filtering

👩‍💻 Author

Sneha Chakraborty
Creative Technologist | ML Enthusiast | Frontend & Creative AI Developer


🌟 Acknowledgements
TensorFlow & Keras
ImageNet
Streamlit
OpenCV
Scikit-learn
VGG16 Architecture Research
💡 Inspiration

ArtWhisper was inspired by the emotional relationship between people and visual art — exploring how machine learning can support more personal and immersive creative discovery experiences.
