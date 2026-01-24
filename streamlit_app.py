import streamlit as st
import requests
from PIL import Image
import io
import json
from pathlib import Path


API_URL = "http://localhost:8000"


st.set_page_config(
    page_title="Recipe Generator",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 3rem;
    }
    .recipe-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .recipe-name {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    .similarity-score {
        font-size: 1rem;
        color: #27AE60;
        font-weight: bold;
    }
    .rank-badge {
        background-color: #FF6B6B;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_recipe(image_file, top_k):
    try:
        files = {"file": ("image.jpg", image_file, "image/jpeg")}
        params = {"top_k": top_k}
        
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")
        return None


def display_recipe(recipe, rank):
    with st.container():
        st.markdown(f"""
            <div class="recipe-card">
                <span class="rank-badge">#{rank}</span>
                <div class="recipe-name">{recipe['name']}</div>
                <div class="similarity-score">🎯 Similarity: {recipe['similarity_score']:.2%}</div>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📝 View Ingredients & Instructions"):
            st.markdown("**Ingredients:**")
            st.write(recipe['ingredients'])
            
            st.markdown("**Instructions:**")
            st.write(recipe['instructions'])


def main():
    st.markdown('<div class="main-header">🍳 AI Recipe Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a food image and discover delicious recipes!</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.warning("Make sure the API is running: `python app.py`")
            st.stop()
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Info")
        st.info("""
        **Vision Model:** ResNet50  
        **Text Model:** MiniLM-L6-v2  
        **Accuracy:** 94.67%
        **Top Results:** 3 Best Matches
        """)
        
        st.markdown("---")
        
        if st.button("🔄 Refresh API Status"):
            st.rerun()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        upload_option = st.radio(
            "Choose upload method:",
            ["Upload from computer", "Use sample image"]
        )
        
        uploaded_file = None
        
        if upload_option == "Upload from computer":
            uploaded_file = st.file_uploader(
                "Choose a food image...",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of food for best results"
            )
        else:
            sample_images = list(Path("artifacts/data/processed/images").glob("*.jpg"))[:5]
            if sample_images:
                selected_sample = st.selectbox(
                    "Select a sample image:",
                    options=[img.name for img in sample_images]
                )
                if selected_sample:
                    with open(f"artifacts/data/processed/images/{selected_sample}", "rb") as f:
                        uploaded_file = f.read()
        
        if uploaded_file:
            try:
                if isinstance(uploaded_file, bytes):
                    image = Image.open(io.BytesIO(uploaded_file))
                else:
                    image = Image.open(uploaded_file)
                
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Predict Recipes", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing your image..."):
                        if isinstance(uploaded_file, bytes):
                            image_bytes = uploaded_file
                        else:
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='JPEG')
                            image_bytes = img_byte_arr.getvalue()
                        
                        result = predict_recipe(image_bytes, 3)
                        
                        if result and result['success']:
                            st.session_state['predictions'] = result['predictions']
                            st.session_state['predicted'] = True
                            st.session_state['top_prediction'] = result['predictions'][0]
                        else:
                            st.error("Prediction failed. Please try again.")
                
                if 'top_prediction' in st.session_state and st.session_state.get('predicted'):
                    st.markdown("---")
                    st.markdown("### 🎯 Best Match Details")
                    top_recipe = st.session_state['top_prediction']
                    
                    st.markdown(f"**{top_recipe['name']}**")
                    st.markdown(f"*Confidence: {top_recipe['similarity_score']:.1%}*")
                    
                    with st.expander("📋 Complete Ingredients List", expanded=True):
                        st.write(top_recipe['ingredients'])
                    
                    with st.expander("👨‍🍳 Cooking Instructions"):
                        st.write(top_recipe['instructions'])
            
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with col2:
        st.header("🍽️ Top 3 Recipe Matches")
        
        if 'predicted' in st.session_state and st.session_state['predicted']:
            predictions = st.session_state['predictions']
            
            st.success(f"✨ Found {len(predictions)} best matching recipes!")
            
            st.markdown("---")
            
            for idx, recipe in enumerate(predictions, 1):
                display_recipe(recipe, idx)
            
            st.markdown("---")
            
            if st.button("🔄 Clear Results"):
                st.session_state['predicted'] = False
                if 'top_prediction' in st.session_state:
                    del st.session_state['top_prediction']
                st.rerun()
            
            if st.download_button(
                label="📥 Download Results (JSON)",
                data=json.dumps(predictions, indent=2),
                file_name="recipe_predictions.json",
                mime="application/json"
            ):
                st.success("Downloaded successfully!")
        
        else:
            st.info("👆 Upload an image and click 'Predict Recipes' to see suggestions!")
            
            st.markdown("### 💡 Tips for Best Results:")
            st.markdown("""
            - Use clear, well-lit food images
            - Focus on the main dish
            - Avoid blurry or dark images
            - Single dish works better than multiple items
            """)
            
            st.markdown("### 🎯 What You'll Get:")
            st.markdown("""
            - **Top 3** most similar recipes
            - **Full ingredients** for the best match
            - **Step-by-step** cooking instructions
            - **Confidence scores** for each prediction
            """)
    
    st.markdown("---")
    
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### 🎯 How It Works
        
        This AI-powered Recipe Generator uses:
        1. **Computer Vision** (ResNet50) to understand food images
        2. **Natural Language Processing** (Sentence Transformers) to understand recipes
        3. **Multimodal Learning** to match images with recipes
        
        ### 📊 Performance
        - **Accuracy:** 94.67%
        - **Recall@5:** 99.57%
        - **Model:** Trained on 4,466 recipes
        
        ### 🚀 Technology Stack
        - **Backend:** FastAPI
        - **Frontend:** Streamlit
        - **ML Framework:** PyTorch
        - **Deployment:** AWS (EC2/Lambda)
        """)


if __name__ == "__main__":
    main()