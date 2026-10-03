import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from google import genai
from google.genai import types
import json
import os

st.set_page_config(page_title="FieldVision AI", layout="centered", initial_sidebar_state="collapsed")

# Custom Field-Optimized Styling
st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; text-align: center; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.95rem; color: #6b7280; text-align: center; margin-bottom: 1.5rem; }
    .badge-occupied { background-color: #dcfce7; color: #15803d; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-vacant { background-color: #fee2e2; color: #b91c1c; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">FieldVision AI Inspector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Photo-to-Data Field Engine</div>', unsafe_allow_html=True)

# 1. Strict Schema mapping directly to mortgage servicing requirements
class InspectionAudit(BaseModel):
    occupancy_status: Literal["Occupied", "Vacant", "Unknown"] = Field(
        description="Determine occupancy status based on visual evidence."
    )
    occupied_by: Literal["Owner", "Tenant", "Vagrant/Squatter", "Unknown"] = Field(
        description="Inferred resident type based on property presentation."
    )
    occupancy_determination_method: Literal["Visual", "Direct contact", "Other"]
    visual_indicators_found: List[str] = Field(
        description="Select all visible indicators: Animals, Car, Decorations, Furniture, Mailbox, Lawn, People"
    )
    property_stories: Literal["1", "2", "3", "4", "5"]
    construction_type: Literal[
        "Brick/Block", "Frame", "Frame and brick", "Stone", "Stucco", "Vinyl frame", "Other"
    ]
    attached_garage_present: bool
    carport_present: bool
    water_features_or_pool_present: bool
    property_for_sale: Literal["For sale by broker", "For sale by owner", "Not for sale"]
    exterior_condition: Literal["Good", "Fair", "Poor"]
    exterior_damage_present: bool
    exterior_damage_details: Optional[str] = Field(
        default="None", description="Specific details of any visible damage to roof, siding, gutters, or windows."
    )
    interior_access_gained: bool = Field(
        description="Set to true ONLY if photos show the inside of interior rooms. If photos show only an exterior door or hardware, set to false."
    )
    occupied_comments: str = Field(
        description="Concise, factual 2-sentence narrative suitable for bank compliance audit."
    )

# 2. Key Management
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if not api_key:
    with st.expander("API Configuration", expanded=True):
        api_key = st.text_input("Enter Gemini API Key", type="password")

# 3. File Capture Interface
uploaded_files = st.file_uploader(
    "Snap or upload 1 to 6 inspection photos", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 4))
    pil_images = []
    for idx, f in enumerate(uploaded_files):
        img = Image.open(f)
        pil_images.append(img)
        with cols[idx % 4]:
            st.image(img, use_container_width=True)

    if st.button("RUN AUDIT (Extract Form Data)", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please provide an API key to run analysis.")
        else:
            with st.spinner("Analyzing photos with AI vision..."):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = (
                        "You are an expert mortgage property field inspector. "
                        "Analyze these photos of the subject property and extract all physical characteristics, "
                        "occupancy indicators, construction styles, and damage. Be strictly factual."
                    )
                    
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[*pil_images, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=InspectionAudit,
                            temperature=0.1
                        )
                    )
                    
                    st.session_state["audit_data"] = json.loads(response.text)
                    st.success("Analysis Complete!")
                except Exception as e:
                    st.error(f"Inference error: {e}")

# 4. Populated Verification Card
if "audit_data" in st.session_state:
    data = st.session_state["audit_data"]
    st.divider()
    
    status_class = "badge-occupied" if data["occupancy_status"] == "Occupied" else "badge-vacant"
    st.markdown(f"### Results: <span class='{status_class}'>{data['occupancy_status'].upper()}</span>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Resident Type:** {data['occupied_by']}")
        st.write(f"**Method:** {data['occupancy_determination_method']}")
        indicators = ", ".join(data["visual_indicators_found"]) if data["visual_indicators_found"] else "None detected"
        st.write(f"**Visual Tags:** {indicators}")
        st.write(f"**Interior Access:** {'YES' if data['interior_access_gained'] else 'NO'}")
        
    with col2:
        st.write(f"**Stories:** {data['property_stories']}")
        st.write(f"**Construction:** {data['construction_type']}")
        st.write(f"**Attached Garage:** {'Yes' if data['attached_garage_present'] else 'No'}")
        st.write(f"**Pool / Water Feature:** {'Yes' if data['water_features_or_pool_present'] else 'No'}")
        st.write(f"**Listing Status:** {data['property_for_sale']}")

    st.write(f"**Condition:** {data['exterior_condition']} | **Damage:** {'YES' if data['exterior_damage_present'] else 'No visible damage'}")
    if data['exterior_damage_present']:
        st.warning(f"**Damage Identified:** {data['exterior_damage_details']}")

    st.info(f"**Bank-Ready Narrative:**\n\n{data['occupied_comments']}")

    if st.button("Confirm & Finalize Inspection", use_container_width=True):
        st.balloons()
        st.success("Inspection ready for PDF packaging!")
