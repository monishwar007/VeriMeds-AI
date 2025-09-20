import streamlit as st
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests
import os
import torch
import io
import base64
import re

# --- Hugging Face Authentication ---
# IMPORTANT: You must have your Hugging Face access token set as an environment variable.
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "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")
os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_TOKEN

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_granite_model():
    """Loads the ibm-granite/granite-docling-258M model and processor."""
    st.write("Loading `ibm-granite/granite-docling-258M` model...")
    processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M", token=HUGGING_FACE_TOKEN)
    model = AutoModelForVision2Seq.from_pretrained("ibm-granite/granite-docling-258M", token=HUGGING_FACE_TOKEN)
    return processor, model

# --- Utility Functions ---
def get_drug_info(drug_name: str):
    """Fetches drug information from a simulated database.

    Behavior:
    - If drug_name is empty/whitespace -> return the full drug database (dict).
    - If drug_name provided -> return the info dict for that drug (or None if not found).
    """
    drug_data = {
        "amoxicillin": {
            "explanation": "Amoxicillin is a broad-spectrum penicillin antibiotic used to treat a variety of bacterial infections, including those of the respiratory tract, urinary tract, and ear. It works by preventing bacteria from building their cell walls.",
            "dosages": {
                "adult": "250-500 mg every 8 hours",
                "child": "25-45 mg/kg/day in divided doses"
            },
            "alternatives": ["Azithromycin", "Doxycycline"],
            "interactions": ["Methotrexate", "Warfarin", "Allopurinol"],
            "side_effects": ["Nausea", "Diarrhea", "Rash"]
        },
        "ibuprofen": {
            "explanation": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain relief, fever reduction, and inflammation. It is commonly used for headaches, menstrual cramps, dental pain, and arthritis.",
            "dosages": {
                "adult": "200-400 mg every 4-6 hours",
                "child": "5-10 mg/kg every 6-8 hours"
            },
            "alternatives": ["Acetaminophen", "Naproxen"],
            "interactions": ["Warfarin", "Aspirin", "Steroids", "ACE inhibitors"],
            "side_effects": ["Stomach pain", "Heartburn", "Dizziness"]
        },
        "warfarin": {
            "explanation": "Warfarin is an anticoagulant (blood thinner) used to prevent blood clots in conditions like atrial fibrillation, DVT, and pulmonary embolism. It requires careful monitoring of INR levels to ensure effectiveness and safety.",
            "dosages": {
                "adult": "2-10 mg/day (adjusted by INR)",
                "child": "0.1 mg/kg/day"
            },
            "alternatives": ["Apixaban", "Dabigatran", "Rivaroxaban"],
            "interactions": ["Aspirin", "Ibuprofen", "Vitamin K-rich foods", "Alcohol"],
            "side_effects": ["Bleeding", "Bruising", "Nausea"]
        },
        "aspirin": {
            "explanation": "Aspirin is an NSAID used for pain relief, fever, and inflammation. In lower doses, it is used as an antiplatelet drug to prevent heart attacks and strokes.",
            "dosages": {
                "adult": "325-650 mg every 4 hours",
                "child": "Not recommended for children under 16 due to risk of Reye's syndrome."
            },
            "alternatives": ["Ibuprofen", "Acetaminophen"],
            "interactions": ["Warfarin", "Ibuprofen", "Alcohol", "Heparin"],
            "side_effects": ["Stomach upset", "Bleeding", "Tinnitus (ringing in ears)"]
        },
        "paracetamol": {
            "explanation": "Paracetamol (Acetaminophen) is an analgesic and antipyretic used for pain relief and fever reduction. It is a common over-the-counter medication.",
            "dosages": {
                "adult": "500-1000 mg every 4-6 hours (max 4 g/day)",
                "child": "10-15 mg/kg every 4-6 hours (max 60 mg/kg/day)"
            },
            "alternatives": ["Ibuprofen", "Naproxen"],
            "interactions": ["Alcohol", "Warfarin (long-term use)"],
            "side_effects": ["Liver damage (with high doses)", "Rash"]
        },
        "metformin": {
            "explanation": "Metformin is an oral antidiabetic medication used to control blood sugar levels in type 2 diabetes. It works by improving insulin sensitivity and reducing glucose production in the liver.",
            "dosages": {
                "adult": "500-2000 mg/day in divided doses",
                "child": "Not commonly used under 10 years"
            },
            "alternatives": ["Sitagliptin", "Glipizide"],
            "interactions": ["Alcohol", "Cimetidine", "Iodinated contrast dyes"],
            "side_effects": ["Diarrhea", "Nausea", "Stomach cramps"]
        },
        "atorvastatin": {
            "explanation": "Atorvastatin is a statin used to lower cholesterol and triglyceride levels. It is prescribed to reduce the risk of heart attack, stroke, and other cardiovascular events.",
            "dosages": {
                "adult": "10-80 mg once daily",
                "child": "Not commonly used under 10 years"
            },
            "alternatives": ["Rosuvastatin", "Simvastatin"],
            "interactions": ["Grapefruit juice", "Warfarin", "Fibrates"],
            "side_effects": ["Muscle pain", "Headache", "Nausea"]
        }
    }

    if not isinstance(drug_name, str) or not drug_name.strip():
        # Return the full database when drug_name is empty or non-string
        return drug_data

    # Return the single drug's info dict (or None if not found)
    return drug_data.get(drug_name.lower())


def check_interactions(drugs):
    """Checks for drug-drug interactions based on the provided list of drugs."""
    found_interactions = []
    lc_drugs = [d.lower() for d in drugs]

    for drug1 in drugs:
        info1 = get_drug_info(drug1)
        if info1 and "interactions" in info1:
            for interaction_drug in info1["interactions"]:
                if interaction_drug.lower() in lc_drugs and drug1.lower() != interaction_drug.lower():
                    # Avoid duplicates
                    pair = tuple(sorted([drug1.lower(), interaction_drug.lower()]))
                    message = f"**{drug1.capitalize()}** and **{interaction_drug.capitalize()}** have a potential interaction. This could increase the risk of side effects or alter the drug's effectiveness. Consult a healthcare professional."
                    if message not in found_interactions:
                        found_interactions.append(message)

    return found_interactions if found_interactions else ["No major drug-drug interactions detected."]


def analyze_age_dosage(age_group, drugs):
    """Provides age-specific dosage recommendations."""
    recommendations = []
    for drug in drugs:
        info = get_drug_info(drug)
        if info:
            if age_group in info.get("dosages", {}):
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
            image_bytes = uploaded_file.getvalue()
            st.image(image_bytes, caption="Uploaded Prescription Image.", use_column_width=True)
            
            prompt = "Extract the full drug names, dosages (e.g., 200mg), and frequency (e.g., once daily) from this medical document. List them in a clear, structured format, like 'Drug: [name], Dosage: [dosage], Frequency: [frequency]'."
            
            with st.spinner("Analyzing image with `ibm-granite` model..."):
                extracted_text = extract_from_image(image_bytes, prompt)
                
            st.subheader("Extracted Prescription Details:")
            st.markdown(extracted_text)
            
            prescription_text = extracted_text
    
    st.markdown("---")
    
    # Analysis & Verification Section
    if st.button("Run Analysis", use_container_width=True, type="primary"):
        if not prescription_text:
            st.error("Please enter prescription text or upload an image to analyze.")
            return

        with st.spinner("Analyzing prescription..."):
            # Safely get the drug database (never None)
            drug_db = get_drug_info('') or {}
            drug_list = list(drug_db.keys())
            drugs_found = []
            for drug in drug_list:
                # Use a case-insensitive regex pattern
                if re.search(r'\b' + re.escape(drug) + r'\b', prescription_text, re.IGNORECASE):
                    drugs_found.append(drug)
            
            drugs = list(set(drugs_found)) # Remove duplicates

            if not drugs:
                st.warning("No recognized drugs found in the provided text.")
            else:
                st.subheader("✅ Verification Results")
                st.write(f"Identified Drugs: {', '.join([d.capitalize() for d in drugs])}")
                st.markdown("---")

                # Drug Information Section (New!)
                st.markdown("<h3 style='color:purple;'>📋 Drug Details and Explanations</h3>", unsafe_allow_html=True)
                for drug in drugs:
                    info = get_drug_info(drug)
                    if info:
                        with st.expander(f"**Learn more about {drug.capitalize()}**"):
                            st.write(f"**Explanation:** {info['explanation']}")
                            if "side_effects" in info:
                                side_effects = ", ".join(info["side_effects"])
                                st.write(f"**Common Side Effects:** {side_effects}")
                            st.write(f"**Known Interactions:** {', '.join(info['interactions'])}")
                            st.write(f"**Alternatives:** {', '.join(info['alternatives'])}")
                    else:
                        st.write(f"**{drug.capitalize()}:** Detailed information not found.")
                st.markdown("---")
                
                # Drug Interaction Detection
                st.markdown("<h3 style='color:red;'>⚠️ Drug Interaction Detection</h3>", unsafe_allow_html=True)
                interactions = check_interactions(drugs)
                for i in interactions:
                    st.markdown(f"- {i}", unsafe_allow_html=True)
                st.markdown("---")
                
                # Age-Specific Dosage Recommendation
                st.markdown("<h3 style='color:blue;'>💊 Age-Specific Dosage Recommendations</h3>", unsafe_allow_html=True)
                dosages = analyze_age_dosage(age_group, drugs)
                for d in dosages:
                    st.markdown(f"- {d}")

                # Alternative Medication Suggestions
                st.markdown("<h3 style='color:green;'>🌿 Alternative Medication Suggestions</h3>", unsafe_allow_html=True)
                alternatives = suggest_alternatives(drugs)
                for a in alternatives:
                    st.markdown(f"- {a}")

    st.markdown("---")
    st.markdown("""
        **Note:** This is a demonstration for educational and development purposes. In a production environment,
        a live, secure database and a robust API would replace the simulated functions.
    """)

if __name__ == "__main__":
    main()
