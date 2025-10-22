# 🔬 Creep Rupture Life Prediction Web App  
🌐 **Live App Link:** [Open on Streamlit Cloud](https://creeppredictor.streamlit.app/)  


This Streamlit tool enables **machine learning–based prediction of creep rupture life** for superalloys based on alloy composition, test conditions, and heat treatment parameters.  

---

## 🚀 Features  
- 🧠 Prediction using two independent ML models: **PyTorch Neural Network** & **AutoGluon Ensemble**  
- ⚙️ Adjustable input parameters for **chemical composition**, **stress**, **temperature**, and **heat treatment**  
- 📈 Model comparison with difference estimation (hours and %)  
- 🧩 Interactive normalization of elemental composition (sum = 100%)  
- 📊 User-friendly sliders with value ranges derived from experimental dataset  
- 💾 Pretrained model loading with automatic feature scaling and inverse transformation  
- 🎨 Clean, responsive UI with custom Streamlit styling

---

# Basic Usage

1. Adjust chemical composition — ensure elements sum to 100% (normalize button included).
2. Set test parameters (stress, temperature) and heat treatment conditions.
3. Choose between AutoML and PyTorch NN or compare both models.
4. Click “🚀 Make Prediction” to compute creep rupture life.



