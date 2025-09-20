import streamlit as st
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests
import os
import torch
import io
import base64

# --- Hugging Face Authentication ---
# IMPORTANT: You must have your Hugging Face access token set as an environment variable.
# For local testing, you can also paste it directly below, but it's not recommended for production.
# The user provided token: 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
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "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")
os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_TOKEN

# --- Model Loading (Cached for performance) ---
# This function will cache the model so it doesn't have to be reloaded on every rerun.
@st.cache_resource
def load_granite_model():
    """Loads the ibm-granite/granite-docling-258M model and processor."""
    st.write("Loading `ibm-granite/granite-docling-258M` model...")
    processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M", token=HUGGING_FACE_TOKEN)
    model = AutoModelForVision2Seq.from_pretrained("ibm-granite/granite-docling-258M", token=HUGGING_FACE_TOKEN)
    return processor, model

# --- Utility Functions ---
# Placeholder for a drug interaction database. In a real-world app, you'd use a
# a licensed database or a robust API like RxNav (with proper handling for its discontinuation).
def get_drug_info(drug_name):
    """Simulates fetching drug information from a database."""
def get_drug_info(drug_name):
    drug_data = {
        "amoxicillin": {
            "explanation": "Amoxicillin is a broad-spectrum penicillin antibiotic used to treat bacterial infections like pneumonia, ear infections, and urinary tract infections.",
            "dosages": {
                "adult": "250-500 mg every 8 hours",
                "child": "25-45 mg/kg/day in divided doses"
            },
            "alternatives": ["Azithromycin", "Doxycycline"],
            "interactions": ["Methotrexate", "Warfarin"]
        },
        "ibuprofen": {
            "explanation": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain relief, fever reduction, and inflammation.",
            "dosages": {
                "adult": "200-400 mg every 4-6 hours",
                "child": "5-10 mg/kg every 6-8 hours"
            },
            "alternatives": ["Acetaminophen", "Naproxen"],
            "interactions": ["Warfarin", "Aspirin", "Steroids"]
        },
        "warfarin": {
            "explanation": "Warfarin is an anticoagulant (blood thinner) used to prevent blood clots in conditions like atrial fibrillation, DVT, and pulmonary embolism.",
            "dosages": {
                "adult": "2-10 mg/day (adjusted by INR)",
                "child": "0.1 mg/kg/day"
            },
            "alternatives": ["Apixaban", "Dabigatran"],
            "interactions": ["Aspirin", "Ibuprofen", "Vitamin K-rich foods"]
        },
        "aspirin": {
            "explanation": "Aspirin is an NSAID used for pain relief, fever, inflammation, and as an antiplatelet drug to prevent heart attacks and strokes.",
            "dosages": {
                "adult": "325-650 mg every 4 hours",
                "child": "Not recommended for children under 16"
            },
            "alternatives": ["Ibuprofen", "Acetaminophen"],
            "interactions": ["Warfarin", "Ibuprofen", "Alcohol"]
        },
        "paracetamol": {
            "explanation": "Paracetamol (Acetaminophen) is an analgesic and antipyretic used for pain relief and fever reduction. It has little anti-inflammatory effect.",
            "dosages": {
                "adult": "500-1000 mg every 4-6 hours (max 4 g/day)",
                "child": "10-15 mg/kg every 4-6 hours (max 60 mg/kg/day)"
            },
            "alternatives": ["Ibuprofen", "Naproxen"],
            "interactions": ["Alcohol", "Warfarin"]
        },
        "metformin": {
            "explanation": "Metformin is an oral antidiabetic medication used to control blood sugar levels in type 2 diabetes by improving insulin sensitivity.",
            "dosages": {
                "adult": "500-2000 mg/day in divided doses",
                "child": "Not commonly used under 10 years"
            },
            "alternatives": ["Sitagliptin", "Glipizide"],
            "interactions": ["Alcohol", "Cimetidine"]
        },
        "atorvastatin": {
            "explanation": "Atorvastatin is a statin used to lower cholesterol and triglyceride levels, reducing the risk of heart attack and stroke.",
            "dosages": {
                "adult": "10-80 mg once daily",
                "child": "Not commonly used under 10 years"
            },
            "alternatives": ["Rosuvastatin", "Simvastatin"],
            "interactions": ["Grapefruit juice", "Warfarin"]
        }
    }

    return drug_data.get(drug_name.lower())


def check_interactions(drugs):
    """Simulates checking for drug-drug interactions."""
    interactions = []
    # Using a simple check for demonstration
    if "warfarin" in drugs and "ibuprofen" in drugs:
        interactions.append("Warfarin and Ibuprofen have a major interaction. Increased risk of bleeding.")
    if "aspirin" in drugs and "warfarin" in drugs:
        interactions.append("Aspirin and Warfarin have a major interaction. Increased risk of bleeding.")
    return interactions if interactions else ["No major drug-drug interactions detected."]

def analyze_age_dosage(age_group, drugs):
    """Provides age-specific dosage recommendations."""
    recommendations = []
    for drug in drugs:
        info = get_drug_info(drug)
        if info:
            if age_group in info["dosages"]:
                recommendations.append(f"**{drug.capitalize()}:** {info['dosages'][age_group]} for a {age_group}.")
            else:
                recommendations.append(f"**{drug.capitalize()}:** Dosage for {age_group} is not available or is not typically used.")
        else:
            recommendations.append(f"**{drug.capitalize()}:** Drug information not found in the database.")
    return recommendations

def suggest_alternatives(drugs):
    """Suggests alternative medications."""
    alternatives = []
    for drug in drugs:
        info = get_drug_info(drug)
        if info and "alternatives" in info:
            alt_list = ", ".join(info["alternatives"])
            alternatives.append(f"**{drug.capitalize()}:** Suggested alternatives are {alt_list}.")
    return alternatives

# --- NLP Processing with ibm-granite/granite-docling-258M ---
def extract_from_image(image_bytes, user_prompt):
    """
    Uses the ibm-granite model to extract structured information from an image.
    The prompt is crucial for getting the desired output format.
    """
    processor, model = load_granite_model()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Correctly encode the image bytes to a base64 string
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        messages = [
            {"role": "user", "content": [
                {"type": "image", "url": f"data:image/jpeg;base64,{image_base64}"},
                {"type": "text", "text": user_prompt}
            ]},
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=250)
        
        # Decode the output, removing the input prompt
        response_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response_text
    
    except Exception as e:
        return f"An error occurred during model inference: {e}"

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="AI Medical Prescription Verifier", layout="wide")

    st.title("AI Medical Prescription Verification")
    st.markdown("""
        This application leverages Hugging Face and other open-source tools to verify medical prescriptions.
        It can analyze text or images to detect drug interactions, provide dosage recommendations, and suggest alternatives.
    """)
    st.markdown("---")

    # Patient Information Section
    with st.expander("📝 Enter Patient and Prescription Details"):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name:")
        with col2:
            patient_age = st.number_input("Patient Age:", min_value=0, max_value=120, value=30)
        
        age_group = "adult" if patient_age >= 18 else "child"
        st.write(f"Patient is in the **{age_group}** age group.")
    
    # Prescription Input
    st.header("Upload Prescription or Enter Manually")
    
    input_method = st.radio("Choose input method:", ["Manual Text Entry", "Upload Prescription Image"])
    
    prescription_text = ""
    if input_method == "Manual Text Entry":
        prescription_text = st.text_area("Enter prescription text here:", height=150)
        
    elif input_method == "Upload Prescription Image":
        uploaded_file = st.file_uploader("Upload an image of the prescription (e.g., JPEG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read image as bytes
            image_bytes = uploaded_file.getvalue()
            st.image(image_bytes, caption="Uploaded Prescription Image.", use_column_width=True)
            
            # Use ibm-granite to extract text
            prompt = "Extract the full drug names, dosages (e.g., 200mg), and frequency (e.g., once daily) from this medical document. List them in a clear, structured format, like 'Drug: [name], Dosage: [dosage], Frequency: [frequency]'."
            
            with st.spinner("Analyzing image with `ibm-granite` model..."):
                extracted_text = extract_from_image(image_bytes, prompt)
                
            st.subheader("Extracted Prescription Details:")
            st.write(extracted_text)
            
            # Simple text parsing for demonstration
            prescription_text = extracted_text
    
    st.markdown("---")
    
    # Analysis & Verification Section
    if st.button("Run Analysis", use_container_width=True, type="primary"):
        if not prescription_text:
            st.error("Please enter prescription text or upload an image to analyze.")
            return

        with st.spinner("Analyzing prescription..."):
            # Simple NLP for drug name extraction from text input
            drugs = [word.lower().strip(",.").replace(" ", "") for word in prescription_text.split() if get_drug_info(word.lower().strip(",."))]

            if not drugs:
                st.warning("No recognized drugs found in the provided text.")
            else:
                st.subheader("✅ Verification Results")
                st.write(f"Identified Drugs: {', '.join([d.capitalize() for d in drugs])}")
                st.markdown("---")

                # Drug Interaction Detection
                st.markdown("<h3 style='color:red;'>⚠️ Drug Interaction Detection</h3>", unsafe_allow_html=True)
                interactions = check_interactions(drugs)
                for i in interactions:
                    st.write(f"- {i}")
                
                # Age-Specific Dosage Recommendation
                st.markdown("<h3 style='color:blue;'>💊 Age-Specific Dosage Recommendations</h3>", unsafe_allow_html=True)
                dosages = analyze_age_dosage(age_group, drugs)
                for d in dosages:
                    st.write(f"- {d}")

                # Alternative Medication Suggestions
                st.markdown("<h3 style='color:green;'>🌿 Alternative Medication Suggestions</h3>", unsafe_allow_html=True)
                alternatives = suggest_alternatives(drugs)
                for a in alternatives:
                    st.write(f"- {a}")

    st.markdown("---")
    st.markdown("""
        **Note:** This is a demonstration for educational and development purposes. In a production environment,
        a live, secure database and a robust API (like one built with FastAPI and IBM Watson) would replace
        the simulated functions and a more sophisticated NLP model would be used.
        The audio-to-text functionality requires a separate library, such as `SpeechRecognition`, which is not included in this single file for simplicity.
        For example: `import speech_recognition as sr` and `r = sr.Recognizer()`
    """)

if __name__ == "__main__":
    main()
