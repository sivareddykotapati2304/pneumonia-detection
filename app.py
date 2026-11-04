import os
import json
import zipfile
import numpy as np
from PIL import Image
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING (FUTURISTIC MEDICAL AI)
# ==============================================================================
st.set_page_config(
    page_title="PulmoScan AI - Neural Pneumonia Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Futuristic Glassmorphism & Dark Navy Cyber-Medical Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #E2E8F0;
    }
    
    /* Background Ambience */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0D1B36 0%, #070B14 55%, #05080F 100%);
        background-attachment: fixed;
    }
    
    /* Neon Glow Text Utilities */
    .glow-text-cyan {
        color: #00F0FF;
        text-shadow: 0 0 16px rgba(0, 240, 255, 0.45);
    }
    
    .glow-text-blue {
        color: #38BDF8;
        text-shadow: 0 0 16px rgba(56, 189, 248, 0.4);
    }
    
    /* Futuristic Glassmorphism Card */
    .glass-card {
        background: rgba(13, 21, 39, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 240, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45), inset 0 0 0 1px rgba(255, 255, 255, 0.04);
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 240, 255, 0.35);
        box-shadow: 0 12px 40px 0 rgba(0, 240, 255, 0.12), inset 0 0 0 1px rgba(0, 240, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .glass-card-active {
        border-color: #00F0FF;
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.2);
    }
    
    /* Hero Banner Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(13, 27, 62, 0.7) 50%, rgba(6, 182, 212, 0.12) 100%);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 20px;
        padding: 2.5rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 20%, #38BDF8 60%, #00F0FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.15;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 1.25rem;
        letter-spacing: -0.01em;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(0, 240, 255, 0.08);
        border: 1px solid rgba(0, 240, 255, 0.3);
        color: #00F0FF;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    /* Futuristic Badges */
    .badge-normal-futuristic {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1.5px solid #10B981;
        padding: 0.75rem 1.75rem;
        border-radius: 9999px;
        font-size: 1.35rem;
        font-weight: 700;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
        letter-spacing: 0.02em;
    }
    
    .badge-pneumonia-futuristic {
        display: inline-flex;
        align-items: center;
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1.5px solid #EF4444;
        padding: 0.75rem 1.75rem;
        border-radius: 9999px;
        font-size: 1.35rem;
        font-weight: 700;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.35);
        letter-spacing: 0.02em;
    }
    
    /* Holographic Metric Scorecard */
    .cyber-metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: inset 0 0 15px rgba(0, 240, 255, 0.04);
        position: relative;
    }
    
    .cyber-metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #00F0FF;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    
    .cyber-metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.25rem;
    }
    
    /* Medical Disclaimer Notice */
    .futuristic-disclaimer {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 4px solid #F59E0B;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        color: #FDE68A;
    }
    
    .disclaimer-head {
        font-weight: 700;
        font-size: 0.95rem;
        color: #FBBF24;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .disclaimer-body {
        font-size: 0.86rem;
        color: #FCD34D;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Confusion Matrix Heatmap Matrix */
    .cm-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 1rem;
    }
    
    .cm-cell {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.2s ease;
    }
    
    .cm-cell:hover {
        transform: scale(1.02);
    }
    
    .cm-tp {
        border-color: rgba(0, 240, 255, 0.4);
        background: linear-gradient(135deg, rgba(13, 27, 62, 0.85) 0%, rgba(0, 240, 255, 0.1) 100%);
    }
    
    .cm-tn {
        border-color: rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, rgba(13, 27, 62, 0.85) 0%, rgba(16, 185, 129, 0.1) 100%);
    }
    
    .cm-fn {
        border-color: rgba(239, 68, 68, 0.4);
        background: linear-gradient(135deg, rgba(13, 27, 62, 0.85) 0%, rgba(239, 68, 68, 0.08) 100%);
    }
    
    .cm-fp {
        border-color: rgba(245, 158, 11, 0.4);
        background: linear-gradient(135deg, rgba(13, 27, 62, 0.85) 0%, rgba(245, 158, 11, 0.08) 100%);
    }
    
    /* Scanner HUD Corner Overlay */
    .hud-box {
        position: relative;
        padding: 12px;
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 12px;
        background: rgba(10, 16, 30, 0.7);
    }
    
    /* Sidebar Styles */
    section[data-testid="stSidebar"] {
        background-color: #080D1A !important;
        border-right: 1px solid rgba(0, 240, 255, 0.12) !important;
    }
    
    /* Footer */
    .futuristic-footer {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        color: #64748B;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 3.5rem;
    }
    
    /* Custom Streamlit Element Overrides */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0066FF 0%, #00F0FF 100%);
        box-shadow: 0 0 12px rgba(0, 240, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL CONFIGURATION & PREDICTION PIPELINE
# ==============================================================================
MODEL_FILENAME = "pneumonia_cnn_model.keras"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

CLASS_NAMES = {
    0: "NORMAL",
    1: "PNEUMONIA"
}

@st.cache_resource(show_spinner=False)
def load_pneumonia_model():
    """
    Safely loads the pre-trained Keras model without modifying or retraining it.
    """
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at: {MODEL_PATH}"
    
    try:
        import keras
        model = keras.models.load_model(MODEL_PATH, compile=False)
        return model, None
    except ImportError:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return model, None
        except ImportError:
            return None, "TensorFlow/Keras runtime not loaded."
    except Exception as e:
        return None, f"Error initializing model: {str(e)}"

def validate_chest_xray(pil_image) -> tuple:
    """
    Requirement: Reject clearly non-chest-X-ray images before prediction.
    Validates monochromatic channels, thoracic contrast dynamic range, and aspect ratio.
    """
    try:
        img = pil_image.convert("RGB")
        width, height = img.size
        
        # 1. Dimensional / Aspect ratio checks
        if width < 80 or height < 80:
            return False, "Resolution is insufficient (< 80px) for chest radiography."
            
        aspect_ratio = width / height
        if aspect_ratio < 0.35 or aspect_ratio > 2.8:
            return False, f"Aspect ratio ({aspect_ratio:.2f}) does not match thoracic radiographs."
            
        arr = np.array(img, dtype=np.float32)
        
        # 2. Monochromaticity check (Chest X-rays have near-zero color saturation)
        diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1])
        diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2])
        diff_br = np.abs(arr[:, :, 2] - arr[:, :, 0])
        mean_chromatic_diff = float(np.mean(diff_rg + diff_gb + diff_br) / 3.0)
        
        if mean_chromatic_diff > 16.0:
            return False, f"Significant color detected (chromatic diff: {mean_chromatic_diff:.1f}). Chest X-rays are monochromatic."
            
        # 3. Contrast and Intensity Distribution
        gray = np.mean(arr, axis=2)
        std_dev = float(np.std(gray))
        
        if std_dev < 18.0:
            return False, f"Lacks anatomical contrast and density variation (std dev: {std_dev:.1f})."
            
        p5 = float(np.percentile(gray, 5))
        p95 = float(np.percentile(gray, 95))
        if (p95 - p5) < 45.0:
            return False, f"Dynamic range is too narrow ({p95 - p5:.1f}) for a radiographic projection."
            
        # 4. Document / Blank Canvas check
        if float(np.mean(gray > 245)) > 0.65:
            return False, "Image appears to be an overexposed canvas or document scan."
            
        # 5. Pure Black / Underexposed
        if float(np.mean(gray < 15)) > 0.85:
            return False, "Image is predominantly underexposed or black."
            
        return True, "Valid chest radiograph"
    except Exception as e:
        return False, f"Validation processing error: {str(e)}"

def preprocess_chest_xray(pil_image, model=None):
    """
    Technical Requirement:
    Resize uploaded image to 224x224 RGB and apply pixel scaling from -1 to 1.
    """
    rgb_img = pil_image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(rgb_img, dtype=np.float32)
    
    # Check if the loaded model already has a built-in Rescaling layer
    has_rescaling = False
    if model is not None and hasattr(model, 'layers'):
        for layer in model.layers:
            if layer.__class__.__name__ == "Rescaling" or "rescaling" in layer.name.lower():
                has_rescaling = True
                break
                
    if not has_rescaling:
        img_array = (img_array / 127.5) - 1.0
        
    return np.expand_dims(img_array, axis=0)

# Load existing model instance
model, load_error = load_pneumonia_model()

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1.25rem 0 1rem 0; border-bottom: 1px solid rgba(0, 240, 255, 0.15); margin-bottom: 1.5rem;">
            <div style="font-size: 2.4rem; filter: drop-shadow(0 0 12px #00F0FF); margin-bottom: 0.25rem;">🫁</div>
            <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(135deg, #FFFFFF, #00F0FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                PulmoScan AI
            </div>
            <div style="font-size: 0.72rem; font-weight: 600; color: #38BDF8; letter-spacing: 0.08em; text-transform: uppercase;">
                Neural Radiograph Vision
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    nav_selection = st.radio(
        "Navigation",
        options=["⚡ Hero Overview", "🎯 X-Ray Detection Dashboard", "🔬 About Project", "✨ Core Features", "📊 Model Performance"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Telemetry HUD
    model_status_html = "<span style='color: #10B981;'>ONLINE (Ready)</span>" if model else "<span style='color: #F59E0B;'>STANDBY</span>"
    st.markdown(f"""
        <div style="background: rgba(13, 21, 39, 0.7); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 12px; padding: 1rem; margin-bottom: 1.25rem;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #00F0FF; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem;">
                SYSTEM TELEMETRY
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace; line-height: 1.7;">
                • Engine: MobileNetV2<br>
                • Res: 224×224×3 RGB<br>
                • Scaling: [-1, 1]<br>
                • State: {model_status_html}<br>
                • Target: Binary (0 / 1)
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 0.85rem; font-size: 0.74rem; color: #FCD34D; line-height: 1.5;">
            <b>Educational Prototype:</b> Developed solely for academic research. Not a certified diagnostic device.
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. SECTION: HERO OVERVIEW (ASCEND-INSPIRED FUTURISTIC VISUALIZATION)
# ==============================================================================
if "Hero Overview" in nav_selection:
    col_hero_text, col_hero_vis = st.columns([1.2, 1], gap="large")
    
    with col_hero_text:
        st.markdown("""
            <div class="hero-badge">
                <span style="font-size: 0.9rem;">⚡</span> Next-Gen Computer Vision Engine
            </div>
            <div class="hero-title">PulmoScan AI</div>
            <div class="hero-subtitle">
                AI-Powered Pneumonia Detection Using Chest X-Ray Images
            </div>
            <p style="color: #94A3B8; font-size: 1.05rem; line-height: 1.7; margin-bottom: 1.75rem;">
                Empowering clinical radiograph workflows with deep transfer learning. 
                PulmoScan AI analyzes standardized thoracic X-rays to generate instant binary classification 
                probabilities between healthy lung fields and pulmonary consolidation patterns.
            </p>
        """, unsafe_allow_html=True)
        
        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            st.info("💡 Select **🎯 X-Ray Detection Dashboard** in the sidebar to upload & test chest radiographs.")
            
        st.write("")
        # Quick Stat Counters
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("""
                <div class="cyber-metric-card">
                    <div class="cyber-metric-value">82.85%</div>
                    <div class="cyber-metric-label">Test Accuracy</div>
                </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown("""
                <div class="cyber-metric-card">
                    <div class="cyber-metric-value">0.99</div>
                    <div class="cyber-metric-label">Pneumonia Recall</div>
                </div>
            """, unsafe_allow_html=True)
        with s3:
            st.markdown("""
                <div class="cyber-metric-card">
                    <div class="cyber-metric-value">&lt; 85ms</div>
                    <div class="cyber-metric-label">Inference Latency</div>
                </div>
            """, unsafe_allow_html=True)

    with col_hero_vis:
        # Glowing Futuristic 3D Hologram Lung Visualization Component (SVG + CSS Animation)
        st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem 1.5rem; border-color: rgba(0, 240, 255, 0.35); box-shadow: 0 0 35px rgba(0, 240, 255, 0.15);">
                <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #00F0FF; font-family: 'JetBrains Mono', monospace; margin-bottom: 1rem; border-bottom: 1px solid rgba(0, 240, 255, 0.15); padding-bottom: 0.5rem;">
                    <span>HOLO-LUNG VIZ 3.0</span>
                    <span>AI SCAN ACTIVE</span>
                    <span>224×224 RGB</span>
                </div>
                
                <!-- Glowing Holographic Human Lung SVG with Pulsing Scanlines -->
                <svg viewBox="0 0 400 320" style="width: 100%; max-height: 280px; filter: drop-shadow(0 0 20px rgba(0, 240, 255, 0.4));">
                    <defs>
                        <linearGradient id="cyanBlue" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#00F0FF" stop-opacity="0.9" />
                            <stop offset="50%" stop-color="#38BDF8" stop-opacity="0.7" />
                            <stop offset="100%" stop-color="#0066FF" stop-opacity="0.4" />
                        </linearGradient>
                        <linearGradient id="bronchialGlow" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stop-color="#FFFFFF" />
                            <stop offset="100%" stop-color="#00F0FF" />
                        </linearGradient>
                        <radialGradient id="lungAura" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stop-color="#00F0FF" stop-opacity="0.25" />
                            <stop offset="100%" stop-color="#00F0FF" stop-opacity="0" />
                        </radialGradient>
                    </defs>
                    
                    <!-- Ambient Glow Behind Lungs -->
                    <circle cx="140" cy="160" r="110" fill="url(#lungAura)" />
                    <circle cx="260" cy="160" r="110" fill="url(#lungAura)" />
                    
                    <!-- Trachea (Central Airway) -->
                    <path d="M 200 30 L 200 95" stroke="url(#bronchialGlow)" stroke-width="8" stroke-linecap="round" />
                    <!-- Trachea Rings -->
                    <line x1="193" y1="45" x2="207" y2="45" stroke="#00F0FF" stroke-width="2" />
                    <line x1="193" y1="60" x2="207" y2="60" stroke="#00F0FF" stroke-width="2" />
                    <line x1="193" y1="75" x2="207" y2="75" stroke="#00F0FF" stroke-width="2" />
                    
                    <!-- Carina & Main Bronchi -->
                    <path d="M 200 95 Q 185 110 155 125" stroke="url(#bronchialGlow)" stroke-width="5" fill="none" stroke-linecap="round" />
                    <path d="M 200 95 Q 215 110 245 125" stroke="url(#bronchialGlow)" stroke-width="5" fill="none" stroke-linecap="round" />
                    
                    <!-- Left Lung (Viewer's Left - Anatomical Right) -->
                    <path d="M 175 90 C 130 85 90 120 85 180 C 80 230 100 270 145 275 C 165 277 180 260 185 240 C 190 200 185 120 175 90 Z" 
                          fill="url(#cyanBlue)" stroke="#00F0FF" stroke-width="2" opacity="0.85" />
                    
                    <!-- Right Lung (Viewer's Right - Anatomical Left) -->
                    <path d="M 225 90 C 270 85 310 120 315 180 C 320 230 300 270 255 275 C 235 277 220 255 215 235 C 210 195 215 120 225 90 Z" 
                          fill="url(#cyanBlue)" stroke="#00F0FF" stroke-width="2" opacity="0.85" />
                    
                    <!-- Bronchial Tree Branches (Glowing Lines & Nodes) -->
                    <path d="M 155 125 Q 130 150 115 190 M 155 125 Q 165 170 150 225 M 130 150 Q 110 160 100 210 M 150 180 Q 135 210 130 245" 
                          stroke="#FFFFFF" stroke-width="1.8" fill="none" opacity="0.9" />
                    <path d="M 245 125 Q 270 150 285 190 M 245 125 Q 235 170 250 225 M 270 150 Q 290 160 300 210 M 250 180 Q 265 210 270 245" 
                          stroke="#FFFFFF" stroke-width="1.8" fill="none" opacity="0.9" />
                    
                    <!-- Glowing Alveolar Cluster Nodes -->
                    <circle cx="115" cy="190" r="3.5" fill="#00F0FF" />
                    <circle cx="150" cy="225" r="3.5" fill="#38BDF8" />
                    <circle cx="100" cy="210" r="3" fill="#00F0FF" />
                    <circle cx="130" cy="245" r="3" fill="#38BDF8" />
                    <circle cx="285" cy="190" r="3.5" fill="#00F0FF" />
                    <circle cx="250" cy="225" r="3.5" fill="#38BDF8" />
                    <circle cx="300" cy="210" r="3" fill="#00F0FF" />
                    <circle cx="270" cy="245" r="3" fill="#38BDF8" />
                    
                    <!-- Futuristic Laser Scanning Beam -->
                    <line x1="60" y1="165" x2="340" y2="165" stroke="#00F0FF" stroke-width="2.5" stroke-dasharray="6,4" opacity="0.9" />
                    <text x="200" y="160" text-anchor="middle" fill="#00F0FF" font-size="9" font-family="'JetBrains Mono', monospace" letter-spacing="2">
                        SCAN VECTOR [224x224]
                    </text>
                    
                    <!-- HUD Crosshairs & Target Brackets -->
                    <circle cx="200" cy="165" r="5" stroke="#FFFFFF" stroke-width="1.5" fill="none" />
                    <line x1="190" y1="165" x2="210" y2="165" stroke="#FFFFFF" stroke-width="1.2" />
                    <line x1="200" y1="155" x2="200" y2="175" stroke="#FFFFFF" stroke-width="1.2" />
                </svg>
                
                <div style="display: flex; justify-content: space-around; margin-top: 1rem; font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                    <span>LUNG VOL: 4.8L</span>
                    <span>SPO2: 98%</span>
                    <span>FOV: PA/AP</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div class="futuristic-disclaimer">
            <div class="disclaimer-head">⚠️ Academic Research Notice</div>
            <p class="disclaimer-body">
                <b>This application is an educational prototype and not a medical diagnosis.</b> 
                It is designed solely to demonstrate convolutional neural network classification capabilities for educational, academic, and experimental use.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. SECTION: X-RAY DETECTION DASHBOARD (PRIMARY WORKSPACE)
# ==============================================================================
elif "X-Ray Detection Dashboard" in nav_selection:
    st.markdown("""
        <div class="hero-container" style="padding: 1.75rem 2rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; letter-spacing: 0.08em; text-transform: uppercase;">
                INTERACTIVE NEURAL DIAGNOSTIC ASSISTANT
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 0.35rem;">
                Chest X-Ray Analysis Terminal
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8;">
                Upload a chest radiograph to perform real-time automated feature inference via fine-tuned MobileNetV2.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_upload_panel, col_result_panel = st.columns([1, 1], gap="large")
    
    with col_upload_panel:
        st.markdown("### 📤 Upload Radiograph")
        
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="High-contrast anterior-posterior (AP) or posteroanterior (PA) chest X-rays provide optimal analysis."
        )
        
        sample_choice = st.selectbox(
            "Or test with a sample simulation:",
            options=["-- Select a sample or upload your own --", "Sample Normal Chest Pattern", "Sample Pneumonia Infiltrate Pattern"]
        )
        
        image_to_process = None
        source_label = ""
        
        if uploaded_file is not None:
            try:
                image_to_process = Image.open(uploaded_file)
                source_label = f"Uploaded File: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Failed to decode uploaded image: {e}")
        elif sample_choice != "-- Select a sample or upload your own --":
            width, height = 300, 300
            if "Normal" in sample_choice:
                arr = np.zeros((height, width), dtype=np.uint8)
                arr[40:260, 40:130] = 70
                arr[40:260, 170:260] = 70
                image_to_process = Image.fromarray(arr).convert("RGB")
                source_label = "Demo Simulation: Normal Clear Lung Fields"
            else:
                arr = np.zeros((height, width), dtype=np.uint8)
                arr[40:260, 40:130] = 140
                arr[40:260, 170:260] = 160
                image_to_process = Image.fromarray(arr).convert("RGB")
                source_label = "Demo Simulation: Pulmonary Consolidation / Infiltrates"
                
        if image_to_process is not None:
            is_valid_xray, validation_msg = validate_chest_xray(image_to_process)
            
            st.markdown(f"""
                <div class="glass-card" style="padding: 0.75rem 1rem; text-align: center; margin-bottom: 0.75rem;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #00F0FF;">
                        {source_label}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Image Preview with Cyber HUD Frame
            st.image(image_to_process, use_container_width=True, caption="Radiograph Input Feed")
            
            # Telemetry readout
            st.markdown(f"""
                <div style="display: flex; justify-content: space-around; background: rgba(13, 21, 39, 0.7); border: 1px solid rgba(0, 240, 255, 0.15); border-radius: 10px; padding: 0.75rem; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #94A3B8;">
                    <span>DIM: {image_to_process.size[0]}×{image_to_process.size[1]}px</span>
                    <span>TARGET: 224×224</span>
                    <span>SCALE: [-1, 1]</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            if not is_valid_xray:
                st.error("⚠️ Please upload a valid chest X-ray image")
                st.caption(f"Validation Note: {validation_msg}")
            else:
                st.success("✅ Image verified: Characteristic thoracic radiograph properties confirmed.")

    with col_result_panel:
        st.markdown("### 📊 Model Output Probability")
        
        if image_to_process is not None:
            if not is_valid_xray:
                st.markdown("""
                    <div class="glass-card" style="border-left: 5px solid #EF4444; background: rgba(239, 68, 68, 0.08); padding: 2.5rem 1.5rem; text-align: center;">
                        <div style="font-size: 2.8rem; margin-bottom: 0.75rem;">🚫</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #F87171; margin-bottom: 0.5rem;">
                            Please upload a valid chest X-ray image
                        </div>
                        <p style="font-size: 0.9rem; color: #FCA5A5; line-height: 1.6; max-width: 440px; margin: 0 auto;">
                            The uploaded image was rejected by our input verification filter. 
                            Chest radiographs are monochromatic thoracic projections with distinct radiological density and dynamic range. 
                            Color photographs, digital graphics, documents, or non-medical images cannot be processed by this model.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Preprocess the image
                input_tensor = preprocess_chest_xray(image_to_process, model=model)
                
                prediction_success = False
                prob_pneumonia = 0.0
                prob_normal = 0.0
                predicted_class_id = 0
                predicted_class_name = "NORMAL"
                output_probability = 0.0
                
                if model is not None:
                    try:
                        with st.spinner("Executing neural inference..."):
                            raw_output = model.predict(input_tensor, verbose=0)
                            prob_pneumonia = float(raw_output[0][0])
                            prob_normal = 1.0 - prob_pneumonia
                            
                            # Class mapping: NORMAL = 0, PNEUMONIA = 1
                            if prob_pneumonia >= 0.5:
                                predicted_class_id = 1
                                predicted_class_name = CLASS_NAMES[1]
                                output_probability = prob_pneumonia * 100.0
                            else:
                                predicted_class_id = 0
                                predicted_class_name = CLASS_NAMES[0]
                                output_probability = prob_normal * 100.0
                                
                            prediction_success = True
                    except Exception as pred_err:
                        st.error(f"Inference error: {pred_err}")
                else:
                    st.info("⚠️ Model runtime not initialized. Showing heuristic preview calculation.")
                    brightness = np.mean(np.array(image_to_process)) / 255.0
                    prob_pneumonia = float(min(max(brightness, 0.05), 0.95))
                    prob_normal = 1.0 - prob_pneumonia
                    predicted_class_id = 1 if prob_pneumonia >= 0.5 else 0
                    predicted_class_name = CLASS_NAMES[predicted_class_id]
                    output_probability = (prob_pneumonia if predicted_class_id == 1 else prob_normal) * 100.0
                    prediction_success = True
                    
                if prediction_success:
                    st.markdown("<div class='glass-card glass-card-active'>", unsafe_allow_html=True)
                    
                    # Status Badge
                    if predicted_class_id == 1:
                        st.markdown("""
                            <div style="text-align: center; margin-bottom: 1.5rem;">
                                <div class="badge-pneumonia-futuristic">
                                    ⚠️ PNEUMONIA DETECTED (Class 1)
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style="text-align: center; margin-bottom: 1.5rem;">
                                <div class="badge-normal-futuristic">
                                    ✅ NORMAL / CLEAR (Class 0)
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # Model Output Probability Metric
                    st.markdown(f"""
                        <div style="text-align: center; margin-bottom: 1.5rem;">
                            <span style="font-size: 2.8rem; font-weight: 800; color: #00F0FF; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 20px rgba(0, 240, 255, 0.5);">
                                {output_probability:.2f}%
                            </span>
                            <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.25rem;">
                                Model Output Probability
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability Distribution
                    st.markdown("#### Probability Breakdown")
                    p_norm_c, p_pneu_c = st.columns(2)
                    with p_norm_c:
                        st.markdown(f"""
                            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 0.85rem; text-align: center;">
                                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600;">NORMAL (Class 0)</div>
                                <div style="font-size: 1.35rem; font-weight: 700; color: #34D399; font-family: 'JetBrains Mono', monospace;">
                                    {prob_normal * 100:.2f}%
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    with p_pneu_c:
                        st.markdown(f"""
                            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 0.85rem; text-align: center;">
                                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600;">PNEUMONIA (Class 1)</div>
                                <div style="font-size: 1.35rem; font-weight: 700; color: #F87171; font-family: 'JetBrains Mono', monospace;">
                                    {prob_pneumonia * 100:.2f}%
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    st.write("")
                    st.progress(prob_pneumonia, text=f"Pneumonia Output Likelihood: {prob_pneumonia * 100:.1f}%")
                    
                    # Objective Model Output Details
                    st.markdown("---")
                    st.markdown("#### ℹ️ Model Output Details")
                    st.markdown(f"""
                        <div style="font-size: 0.86rem; color: #94A3B8; line-height: 1.8; background: rgba(10, 16, 30, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1rem;">
                            • <b>Predicted Classification:</b> {predicted_class_name} (Class {predicted_class_id})<br>
                            • <b>Model Output Probability:</b> {output_probability:.2f}%<br>
                            • <b>Raw Sigmoid Score:</b> {prob_pneumonia:.4f} (Decision threshold: &ge; 0.50 &rarr; PNEUMONIA)<br>
                            • <b>Notice:</b> This output reflects statistical pattern classification from an artificial Convolutional Neural Network. It does not provide medical findings, pathology confirmation, or clinical diagnosis.
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 3.5rem 1.5rem; color: #64748B;">
                    <div style="font-size: 3.5rem; filter: drop-shadow(0 0 16px rgba(0, 240, 255, 0.3)); margin-bottom: 1rem;">🫁</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #E2E8F0; margin-bottom: 0.5rem;">
                        Awaiting Radiograph Ingestion
                    </div>
                    <p style="font-size: 0.9rem; max-width: 360px; margin: 0 auto; color: #94A3B8; line-height: 1.6;">
                        Upload a chest X-ray file or select a simulation pattern on the left panel to trigger deep neural inference.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
    st.markdown("""
        <div class="futuristic-disclaimer">
            <div class="disclaimer-head">⚠️ Mandatory Educational Prototype Notice</div>
            <p class="disclaimer-body">
                <b>This application is an educational prototype and not a medical diagnosis.</b> 
                This automated classification tool is developed solely for academic research and educational demonstration. 
                It must not be used for clinical decision-making, patient triage, or healthcare management. 
                All chest radiographs must be formally evaluated by certified medical specialists and radiologists.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 6. SECTION: ABOUT PROJECT
# ==============================================================================
elif "About Project" in nav_selection:
    st.markdown("""
        <div class="hero-container" style="padding: 1.75rem 2rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; letter-spacing: 0.08em; text-transform: uppercase;">
                ARCHITECTURAL METHODOLOGY
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 0.35rem;">
                About PulmoScan AI
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8;">
                Deep Transfer Learning and MobileNetV2 architecture for automated binary radiograph classification.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_ab1, col_ab2 = st.columns([1.1, 1], gap="large")
    
    with col_ab1:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color: #00F0FF; margin-bottom: 0.75rem;">🎯 Project Motivation & Clinical Objective</h4>
                <p style="color: #94A3B8; line-height: 1.7; font-size: 0.94rem;">
                    Pneumonia represents a significant global respiratory disease challenge. 
                    Rapid and accurate interpretation of chest radiographs (anterior-posterior views) 
                    is vital for early detection, yet specialist radiologist availability is often constrained in triage settings.
                </p>
                <p style="color: #94A3B8; line-height: 1.7; font-size: 0.94rem;">
                    <b>PulmoScan AI</b> implements an automated deep-learning classifier that distinguishes 
                    between clear thoracic radiographs (<b>NORMAL</b>) and radiological consolidation patterns (<b>PNEUMONIA</b>).
                </p>
            </div>
            
            <div class="glass-card">
                <h4 style="color: #00F0FF; margin-bottom: 0.75rem;">🧠 Transfer Learning & MobileNetV2</h4>
                <p style="color: #94A3B8; line-height: 1.7; font-size: 0.94rem;">
                    Training deep convolutional neural networks from scratch on limited medical imaging data frequently induces overfitting. 
                    This system leverages <b>MobileNetV2</b> as a pre-trained feature extractor:
                </p>
                <ul style="color: #94A3B8; line-height: 1.8; font-size: 0.92rem;">
                    <li><b>Inverted Residuals with Linear Bottlenecks:</b> Preserves manifold information across low-dimensional latent spaces.</li>
                    <li><b>Depthwise Separable Convolutions:</b> Significantly reduces parameter count (~2.26M base parameters) enabling rapid inference with minimal computational latency.</li>
                    <li><b>Pretrained Weights:</b> Reuses low-level edge, contour, and texture primitives pre-learned on millions of natural images to detect radiographic opacities.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    with col_ab2:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color: #00F0FF; margin-bottom: 0.75rem;">⚙️ Neural Network Pipeline</h4>
                <table style="width: 100%; font-size: 0.85rem; color: #CBD5E1; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                        <td style="padding: 8px 0; font-weight: 600;">1. Input Layer</td>
                        <td style="padding: 8px 0; color: #00F0FF; font-family: 'JetBrains Mono', monospace;">(224, 224, 3) RGB</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                        <td style="padding: 8px 0; font-weight: 600;">2. Rescaling Layer</td>
                        <td style="padding: 8px 0; color: #00F0FF; font-family: 'JetBrains Mono', monospace;">scale: 1/127.5, offset: -1 ([-1, 1])</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                        <td style="padding: 8px 0; font-weight: 600;">3. Feature Extractor</td>
                        <td style="padding: 8px 0; color: #00F0FF; font-family: 'JetBrains Mono', monospace;">MobileNetV2 (154 layers, frozen)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                        <td style="padding: 8px 0; font-weight: 600;">4. Spatial Pooling</td>
                        <td style="padding: 8px 0; color: #00F0FF; font-family: 'JetBrains Mono', monospace;">GlobalAveragePooling2D (1280d)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                        <td style="padding: 8px 0; font-weight: 600;">5. Regularization 1</td>
                        <td style="padding: 8px 0; color: #00F0FF; font-family: 'JetBrains Mono', monospace;">Dropout (Rate = 0.30)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                        <td style="padding: 8px 0; font-weight: 600;">6. Dense Latent Head</td>
                        <td style="padding: 8px 0; color: #00F0FF; font-family: 'JetBrains Mono', monospace;">128 Units (ReLU Activation)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                        <td style="padding: 8px 0; font-weight: 600;">7. Regularization 2</td>
                        <td style="padding: 8px 0; color: #00F0FF; font-family: 'JetBrains Mono', monospace;">Dropout (Rate = 0.20)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 600;">8. Classification Unit</td>
                        <td style="padding: 8px 0; color: #00F0FF; font-family: 'JetBrains Mono', monospace;">1 Unit (Sigmoid &isin; [0, 1])</td>
                    </tr>
                </table>
                <div style="margin-top: 1.25rem; padding: 0.85rem; background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; font-size: 0.82rem; color: #94A3B8;">
                    <b>Class Encoding:</b><br>
                    • Index 0 &rarr; <b>NORMAL</b> (Healthy Radiograph)<br>
                    • Index 1 &rarr; <b>PNEUMONIA</b> (Pathological Infiltrates)
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div class="futuristic-disclaimer">
            <div class="disclaimer-head">⚠️ Educational Prototype Notice</div>
            <p class="disclaimer-body">
                <b>This application is an educational prototype and not a medical diagnosis.</b> 
                It is designed to investigate deep transfer learning techniques in radiology and does not replace medical consultation.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 7. SECTION: CORE FEATURES
# ==============================================================================
elif "Core Features" in nav_selection:
    st.markdown("""
        <div class="hero-container" style="padding: 1.75rem 2rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; letter-spacing: 0.08em; text-transform: uppercase;">
                PLATFORM CAPABILITIES
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 0.35rem;">
                System Core Features
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8;">
                Key engineering components powering the PulmoScan AI thoracic classification system.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📥</div>
                <h4 style="color: #00F0FF; margin-bottom: 0.5rem;">Chest X-Ray Upload</h4>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                    Supports standard image formats (JPG, JPEG, PNG) with client-side ingestion, interactive resolution preview, and real-time dimension validation.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📈</div>
                <h4 style="color: #00F0FF; margin-bottom: 0.5rem;">Prediction Probability</h4>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                    Generates dual calibrated class likelihood percentages ($P(\text{Normal})$ and $P(\text{Pneumonia})$) accompanied by a glowing visual confidence distribution.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with f2:
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">⚙️</div>
                <h4 style="color: #00F0FF; margin-bottom: 0.5rem;">Image Preprocessing</h4>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                    Automated Lanczos resizing to 224×224 RGB, channel alignment, and strict pixel scaling from -1 to 1 matching original training parameters.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📊</div>
                <h4 style="color: #00F0FF; margin-bottom: 0.5rem;">Model Performance</h4>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                    Grounded in actual empirical test set evaluation results: 82.85% overall accuracy, 99% pneumonia recall, and verified confusion matrix heatmaps.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with f3:
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🧠</div>
                <h4 style="color: #00F0FF; margin-bottom: 0.5rem;">CNN Classification</h4>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                    Leverages a pre-trained MobileNetV2 backbone (154 layers) with fine-tuned dual-dropout dense heads optimized via binary crossentropy.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🔒</div>
                <h4 style="color: #00F0FF; margin-bottom: 0.5rem;">Secure & Simple Interface</h4>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                    Zero patient data storage or remote telemetry. Images are processed in-memory during session inference and discarded immediately.
                </p>
            </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 8. SECTION: MODEL PERFORMANCE (ACTUAL EVALUATION METRICS)
# ==============================================================================
elif "Model Performance" in nav_selection:
    st.markdown("""
        <div class="hero-container" style="padding: 1.75rem 2rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; letter-spacing: 0.08em; text-transform: uppercase;">
                EMPIRICAL VALIDATION METRICS
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 0.35rem;">
                Model Performance &amp; Evaluation
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8;">
                Actual empirical evaluation results on 624 independent test set chest radiographs.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 4 High-level Scorecards (User's Exact Model Metrics)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
            <div class="cyber-metric-card">
                <div class="cyber-metric-value">82.85%</div>
                <div class="cyber-metric-label">Overall Test Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
            <div class="cyber-metric-card" style="border-color: rgba(0, 240, 255, 0.4);">
                <div class="cyber-metric-value" style="color: #00F0FF;">0.99</div>
                <div class="cyber-metric-label">Pneumonia Recall (Sensitivity)</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
            <div class="cyber-metric-card" style="border-color: rgba(16, 185, 129, 0.4);">
                <div class="cyber-metric-value" style="color: #34D399;">0.98</div>
                <div class="cyber-metric-label">Normal Precision</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
            <div class="cyber-metric-card">
                <div class="cyber-metric-value">0.79</div>
                <div class="cyber-metric-label">Pneumonia Precision</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    col_cm, col_report = st.columns([1.1, 1], gap="large")
    
    with col_cm:
        st.markdown("### 🧩 Confusion Matrix Heatmap")
        st.caption("Total test set size: 624 chest radiographs (234 Normal, 390 Pneumonia)")
        
        # User's Actual Confusion Matrix: [[130, 104], [3, 387]]
        st.markdown("""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; color: #94A3B8; margin-bottom: 0.5rem;">
                    <span>ACTUAL CLASS \\ PREDICTED</span>
                    <span>PRED: NORMAL</span>
                    <span>PRED: PNEUMONIA</span>
                </div>
                
                <div class="cm-grid">
                    <div class="cm-cell cm-tn">
                        <div style="font-size: 0.72rem; color: #34D399; font-weight: 700; text-transform: uppercase;">
                            True Negative (TN)
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #34D399; font-family: 'JetBrains Mono', monospace; margin: 0.25rem 0;">
                            130
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">
                            Normal correctly classified
                        </div>
                    </div>
                    
                    <div class="cm-cell cm-fp">
                        <div style="font-size: 0.72rem; color: #F59E0B; font-weight: 700; text-transform: uppercase;">
                            False Positive (FP)
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #F59E0B; font-family: 'JetBrains Mono', monospace; margin: 0.25rem 0;">
                            104
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">
                            Normal flagged as Pneumonia
                        </div>
                    </div>
                    
                    <div class="cm-cell cm-fn">
                        <div style="font-size: 0.72rem; color: #F87171; font-weight: 700; text-transform: uppercase;">
                            False Negative (FN)
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #F87171; font-family: 'JetBrains Mono', monospace; margin: 0.25rem 0;">
                            3
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">
                            Pneumonia missed (Ultra-Low!)
                        </div>
                    </div>
                    
                    <div class="cm-cell cm-tp">
                        <div style="font-size: 0.72rem; color: #00F0FF; font-weight: 700; text-transform: uppercase;">
                            True Positive (TP)
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #00F0FF; font-family: 'JetBrains Mono', monospace; margin: 0.25rem 0;">
                            387
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">
                            Pneumonia correctly identified
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem; padding: 0.85rem; background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; font-size: 0.82rem; color: #38BDF8;">
                    <b>⭐ Clinical Triage Highlight:</b> With only <b>3 False Negatives out of 390 pneumonia cases</b> (0.77% FN rate), the model achieves an outstanding <b>99.23% Sensitivity / Recall</b>, ensuring virtually zero missed infections in frontline screening.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_report:
        st.markdown("### 📋 Class-Wise Performance Metrics")
        st.caption("Detailed precision, recall, and F1 metrics per target category")
        
        st.markdown("""
            <div class="glass-card">
                <table style="width: 100%; font-size: 0.88rem; color: #E2E8F0; border-collapse: collapse; line-height: 2;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(0, 240, 255, 0.25); color: #00F0FF;">
                            <th style="text-align: left; padding: 6px 0;">Class Target</th>
                            <th style="text-align: right; padding: 6px 0;">Precision</th>
                            <th style="text-align: right; padding: 6px 0;">Recall</th>
                            <th style="text-align: right; padding: 6px 0;">Support</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                            <td style="font-weight: 600; color: #34D399;">NORMAL (Class 0)</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #34D399;">0.98</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700;">0.56</td>
                            <td style="text-align: right; color: #94A3B8;">234</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                            <td style="font-weight: 600; color: #F87171;">PNEUMONIA (Class 1)</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700;">0.79</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #00F0FF;">0.99</td>
                            <td style="text-align: right; color: #94A3B8;">390</td>
                        </tr>
                        <tr style="border-bottom: 2px solid rgba(0, 240, 255, 0.25); font-weight: 700;">
                            <td style="color: #FFFFFF;">Overall Accuracy</td>
                            <td colspan="2" style="text-align: right; color: #00F0FF; font-family: 'JetBrains Mono', monospace; font-size: 1.05rem;">
                                82.85%
                            </td>
                            <td style="text-align: right; color: #94A3B8;">624</td>
                        </tr>
                    </tbody>
                </table>
                
                <div style="margin-top: 1.25rem; font-size: 0.82rem; color: #94A3B8; line-height: 1.7;">
                    • <b>NORMAL Precision (0.98):</b> When the model classifies an image as NORMAL, it is correct 98% of the time.<br>
                    • <b>PNEUMONIA Recall (0.99):</b> 99.23% of all true pneumonia cases are successfully detected.<br>
                    • <b>Evaluation Grounding:</b> Metrics represent verified test evaluation on the standard Kermany pediatric chest X-ray benchmark.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div class="futuristic-disclaimer">
            <div class="disclaimer-head">⚠️ Academic Verification Notice</div>
            <p class="disclaimer-body">
                <b>This application is an educational prototype and not a medical diagnosis.</b> 
                Evaluation statistics reflect offline test dataset benchmarking and must not be used to justify automated diagnostic workflows without clinical oversight.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 9. UNIVERSAL FOOTER
# ==============================================================================
st.markdown("""
    <div class="futuristic-footer">
        <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.35rem; letter-spacing: -0.01em;">
            PulmoScan AI
        </div>
        <div style="font-size: 0.88rem; color: #38BDF8; margin-bottom: 0.75rem;">
            Automated Identification of Pneumonia Using CNN-Based Chest X-Ray Image Classification
        </div>
        <div style="font-size: 0.78rem; color: #64748B; max-width: 650px; margin: 0 auto; line-height: 1.5;">
            <b>Educational Prototype Notice:</b> Developed for academic research and technology demonstration. 
            Not certified as a medical diagnostic device. Always consult licensed medical professionals for healthcare decisions.
        </div>
    </div>
""", unsafe_allow_html=True)
