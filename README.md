# PulmoScan AI - Pneumonia Detection Dashboard

### Automated Identification of Pneumonia Using CNN-Based Chest X-Ray Image Classification

An interactive clinical assistive web dashboard developed with **Python**, **Streamlit**, and **TensorFlow / Keras**.
Utilizes **Transfer Learning** with a fine-tuned **MobileNetV2** deep convolutional neural network backbone to classify chest radiographs into:
- **0: NORMAL**
- **1: PNEUMONIA**

---

### Features
- 🩺 **Real-time Diagnostic Assistant**: Upload chest radiographs (JPG/PNG) for automated inference.
- 📐 **Standardized Preprocessing**: Automatic $224 \times 224$ RGB resizing and $[-1, 1]$ pixel normalization.
- 📊 **Probability & Confidence Distribution**: Live likelihood visualization and clinical findings interpretation.
- 🔬 **Verified Architecture**: MobileNetV2 base (154 layers, 2.26M frozen weights) + custom classifier head (164,097 trainable weights).
- ⚠️ **Medical Disclaimer**: Clear educational notice stating that this system is a technical prototype and not a clinical diagnosis.

---

### Running Locally
```bash
git clone https://github.com/sivareddykotapati2304/pneumonia-detection.git
cd pneumonia-detection
pip install -r requirements.txt
streamlit run app.py
```
