import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from google import genai
from google.genai import types
import json
import os
import time

st.set_page_config(page_title="FieldVision AI", layout="centered", initial_sidebar_state="collapsed")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; text-align: center; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.95rem; color: #6b7280; text-align: center; margin-bottom: 1.5rem; }
    .badge-occupied { background-color: #dcfce7; color: #15803d; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-vacant { background-color: #fee2e2; color: #b91c1c; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .photo-tag { 
        background-color: #0f172a; 
        color: #f8fafc; 
        font-size: 0.70rem; 
        font-weight: 600; 
        text-align: center; 
        padding: 4px 6px; 
        border-radius: 4px; 
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">FieldVision AI Inspector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Photo-to-Data Field Engine</div>', unsafe_allow_html=True)

PhotoLabel = Literal[
    "Street Sign", 
    "Street Scene", 
    "Front Elevation", 
    "House Number / Address", 
    "Lockbox",
    "Vacancy / Preservation Notice",
    "Mailbox / Mail Overflow",
    "Left Side", 
    "Right Side", 
    "Rear Elevation / Deck", 
    "HVAC / Compressor",
    "Electric Meter / Utility",
    "Gas Meter / Utility",
    "Door Lock / Keyway",
    "Interior Thru Window (Bare/Empty)",
    "Interior - Foyer / Entry",
    "Interior - Living Room",
    "Interior - Fireplace Detail",
    "Interior - Kitchen",
    "Interior - Bedroom",
    "Interior - Bathroom",
    "Interior - Basement / Mechanical",
    "Interior - Debris / Damage",
    "Roof Detail", 
    "Lawn / Yard", 
    "Damage Detail", 
    "Other / Unclassified"
]

class InspectionAudit(BaseModel):
    photo_classifications: List[PhotoLabel] = Field(
        description="The exact visual label for each uploaded image in the exact order received."
    )
    occupancy_status: Literal["Occupied", "Vacant", "Unknown"] = Field(
        description="Must be Vacant if lockbox, servicer posting, or bare unfurnished interior is observed."
    )
    occupied_by: Literal["Owner", "Tenant", "Vacant/None", "Vagrant/Squatter", "Unknown"]
    occupancy_determination_method: Literal["Visual", "Direct contact", "Other"]
    visual_indicators_found: List[str] = Field(
        description="Detected tags: Empty Interior, Lockbox, Posting/Sticker, Mail Overflow, Mowed Lawn, Electric Meter Present, HVAC Intact, Personal Property, Debris"
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
    exterior_damage_details: Optional[str] = Field(default="None")
    interior_access_gained: bool = Field(
        description="True if photographs were captured inside the living spaces/rooms. False if curb/window only."
    )
    interior_condition: Literal["Good", "Fair", "Poor", "Not Inspected"]
    interior_debris_present: bool
    electric_meter_installed: bool
    hvac_exterior_unit_present: bool
    occupied_comments: str = Field(
        description="Factual 2-3 sentence mortgage audit summary highlighting exterior vacancy markers, utility presence, and empty interior status."
    )

api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if not api_key:
    with st.expander("API Configuration", expanded=True):
        api_key = st.text_input("Enter Gemini API Key", type="password")

uploaded_files = st.file_uploader(
    "Snap or upload inspection photos (exterior, utilities, and interior)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}"
)

if uploaded_files:
    pil_images = [Image.open(f) for f in uploaded_files]

    if "audit_data" not in st.session_state:
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, img in enumerate(pil_images):
            with cols[idx % 4]:
                st.image(img, use_container_width=True)

    if st.button("RUN AUDIT (Extract Form Data)", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please provide an API key to run analysis.")
        else:
            with st.spinner("Classifying comprehensive photo pack and auditing property..."):
                client = genai.Client(api_key=api_key)
                prompt = (
                    "You are an expert mortgage field inspector analyzing an entire photo pack for default loan servicing.\n\n"
                    "RULES:\n"
                    "1. Classify EVERY uploaded image in order. Use precise tags such as: Street Sign, Street Scene, Front Elevation, House Number / Address, "
                    "Lockbox, Vacancy / Preservation Notice, Rear Elevation / Deck, Left Side, Right Side, HVAC / Compressor, Electric Meter / Utility, "
                    "Door Lock / Keyway, Interior - Foyer / Entry, Interior - Living Room, Interior - Fireplace Detail, Interior - Kitchen, Interior - Bathroom, etc.\n"
                    "2. OCCUPANCY: If you observe a lockbox, servicer vacancy posting, or empty interior rooms without furniture, the status MUST be VACANT.\n"
                    "3. ACCESS: If interior rooms are photographed from inside the dwelling, mark interior_access_gained = True.\n"
                    "4. UTILITIES & ASSETS: Note whether the exterior HVAC/compressor and electric meter are installed and intact."
                )

                candidate_models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
                audit_success = False
                last_err = ""

                for mod in candidate_models:
                    for attempt in range(2):
                        try:
                            response = client.models.generate_content(
                                model=mod,
                                contents=[*pil_images, prompt],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=InspectionAudit,
                                    temperature=0.1
                                )
                            )
                            st.session_state["audit_data"] = json.loads(response.text)
                            st.session_state["cached_images"] = pil_images
                            audit_success = True
                            break
                        except Exception as e:
                            last_err = str(e)
                            time.sleep(1.0)
                    if audit_success:
                        break

                if audit_success:
                    st.rerun()
                else:
                    st.error(f"Inference error: {last_err}")

if "audit_data" in st.session_state and "cached_images" in st.session_state:
    data = st.session_state["audit_data"]
    images = st.session_state["cached_images"]
    labels = data.get("photo_classifications", [])

    st.markdown("### Labeled Photo Evidence")
    labeled_cols = st.columns(min(len(images), 4))
    for idx, img in enumerate(images):
        with labeled_cols[idx % 4]:
            st.image(img, use_container_width=True)
            lbl = labels[idx] if idx < len(labels) else "Photo"
            st.markdown(f'<div class="photo-tag">{lbl}</div>', unsafe_allow_html=True)

    st.divider()

    status_class = "badge-occupied" if data["occupancy_status"] == "Occupied" else "badge-vacant"
    st.markdown(f"### Audit Verdict: <span class='{status_class}'>{data['occupancy_status'].upper()}</span>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Resident Type:** {data['occupied_by']}")
        st.write(f"**Method:** {data['occupancy_determination_method']}")
        indicators = ", ".join(data["visual_indicators_found"]) if data["visual_indicators_found"] else "None detected"
        st.write(f"**Visual Tags:** {indicators}")
        st.write(f"**Interior Entry Gained:** {'YES' if data['interior_access_gained'] else 'NO'}")
        st.write(f"**Interior Condition:** {data['interior_condition']}")
        st.write(f"**Interior Debris/Hazards:** {'YES' if data['interior_debris_present'] else 'None'}")
        
    with col2:
        st.write(f"**Stories:** {data['property_stories']}")
        st.write(f"**Construction:** {data['construction_type']}")
        st.write(f"**Attached Garage:** {'Yes' if data['attached_garage_present'] else 'No'}")
        st.write(f"**HVAC Compressor Intact:** {'Yes' if data.get('hvac_exterior_unit_present') else 'No'}")
        st.write(f"**Electric Meter Present:** {'Yes' if data.get('electric_meter_installed') else 'No'}")
        st.write(f"**Listing Status:** {data['property_for_sale']}")

    st.write(f"**Exterior Condition:** {data['exterior_condition']} | **Damage:** {'YES' if data['exterior_damage_present'] else 'No visible damage'}")
    if data['exterior_damage_present']:
        st.warning(f"**Damage Identified:** {data['exterior_damage_details']}")

    st.info(f"**Bank-Ready Narrative:**\n\n{data['occupied_comments']}")

    if st.button("Clear / Next Inspection", use_container_width=True):
        st.session_state["uploader_key"] += 1
        st.session_state.pop("audit_data", None)
        st.session_state.pop("cached_images", None)
        st.rerun()
