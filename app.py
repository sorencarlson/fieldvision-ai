import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from google import genai
from google.genai import types
import json
import os
import time

st.set_page_config(page_title="FieldVision AI Pro", layout="wide", initial_sidebar_state="collapsed")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; text-align: center; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.92rem; color: #6b7280; text-align: center; margin-bottom: 1.2rem; }
    .badge-occupied { background-color: #dcfce7; color: #15803d; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 1.05rem; }
    .badge-vacant { background-color: #fee2e2; color: #b91c1c; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 1.05rem; }
    .photo-tag { 
        background-color: #0f172a; 
        color: #f8fafc; 
        font-size: 0.68rem; 
        font-weight: 600; 
        text-align: center; 
        padding: 5px 6px; 
        border-radius: 4px; 
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">FieldVision AI Inspector Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Photo-to-Data Engine • InspectorADE Standard</div>', unsafe_allow_html=True)

# 1. InspectorADE Standard Taxonomy
PhotoLabel = Literal[
    # Access & Location Identifiers
    "Street Sign", "Street Scene", "Front Yard", "House Number / Address",
    "Property to Left", "Property to Right", "Foundation", "Roof Condition", "Roof Damage",
    
    # Vacancy, Notices & Security (ADE Taxonomy)
    "Lockbox", "Missing Lockbox", "Vacant Sticker", "Posting", "Through Window",
    "Doors Need Securing", "Windows Boarded", "Windows Broken", "Unable to access interior",
    "No Trespassing", "Key Working",
    
    # Utilities & Mechanicals (ADE Taxonomy)
    "Electric Meter Location", "Electric Meter Location Missing",
    "Water Meter", "Water Shutoff", "Water Tank", "Water Heater Location", "Water Heater Location Missing",
    "Open Breaker Box", "Furnace", "Sump Pump", "Propane Tank", "Oil Tank", "Volt Stick",
    
    # Interior Rooms & Zones (ADE Taxonomy)
    "Foyer", "Living Room", "Living Room Condition", "Family Room", "Kitchen", "Kitchen Condition",
    "Master Bathroom", "Half Bathroom", "Utility Room", "Sun Porch", "Stairway Condition", "Hallway Condition",
    
    # Hazards, Debris & Property Status (ADE Taxonomy)
    "Interior Debris", "Exterior Debris", "Interior Health Hazard", "Exterior Health Hazard",
    "Interior Personal Property", "Exterior Personal Property", "Vandalism", "Water Damage",
    "Freeze Damage", "Fire Damage", "Mold", "Mortgagor Neglect", "Wear and Tear",
    "Gutters/Downspouts Damaged", "Handrails Damages/Missing", "Holes/Trip Hazards",
    "Outbuilding", "Outbuilding Condition", "Garage", "Garage Condition", "Fence",
    "Pool Condition", "Pool fence/gate/lanai", "VIN#/Plate", "Yard Condition",
    "Other / Unclassified"
]

class InspectionAudit(BaseModel):
    photo_classifications: List[PhotoLabel] = Field(
        description="The exact InspectorADE visual label for each image in the order received."
    )
    occupancy_status: Literal["Occupied", "Vacant", "Unknown"] = Field(
        description="Must be Vacant if Lockbox, Vacant Sticker, Posting, Through Window (bare interior), or bare rooms are present. A cut lawn does not prove occupancy."
    )
    occupied_by: Literal["Owner", "Tenant", "Vacant/None", "Vagrant/Squatter", "Unknown"]
    occupancy_determination_method: Literal["Visual", "Direct contact", "Other"]
    visual_indicators_found: List[str] = Field(
        description="All observed indicators matching ADE tags: Lockbox, Vacant Sticker, Posting, Through Window, Electric Meter, HVAC, Yard Maintained, Personal Property, Debris."
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
        description="True ONLY if actual entry was made into the home. False for exterior shots or Through Window peeks."
    )
    interior_condition: Literal["Good", "Fair", "Poor", "Not Inspected"]
    interior_debris_present: bool
    electric_meter_installed: bool
    occupied_comments: str = Field(
        description="Bank-compliant 2-3 sentence mortgage field narrative detailing occupancy evidence (lockbox, postings, window peeks) and observed physical condition."
    )

# 2. Key Management
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if not api_key:
    with st.expander("API Configuration", expanded=True):
        api_key = st.text_input("Enter Gemini API Key", type="password")

# 3. Photo Capture
uploaded_files = st.file_uploader(
    "Upload Inspection Photo Pack (No Limit)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}"
)

if uploaded_files:
    if "cached_images" not in st.session_state:
        st.session_state["cached_images"] = [Image.open(f) for f in uploaded_files]

    images = st.session_state["cached_images"]

    if "audit_data" not in st.session_state:
        st.caption(f"Loaded {len(images)} inspection photos ready for ADE tagging.")
        cols = st.columns(min(len(images), 6))
        for idx, img in enumerate(images):
            with cols[idx % 6]:
                st.image(img, use_container_width=True)

        if st.button(f"RUN INSPECTOR-ADE AUDIT ({len(images)} PHOTOS)", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please provide an API key.")
            else:
                with st.spinner("Applying InspectorADE classification and auditing property..."):
                    client = genai.Client(api_key=api_key)
                    prompt = (
                        "You are an expert mortgage field inspector labeling photo packs for default servicing using InspectorADE standards.\n\n"
                        "STRICT RULES:\n"
                        "1. CLASSIFICATION: For every image in exact sequential order, select the matching label from PhotoLabel "
                        "(e.g., Lockbox, Vacant Sticker, Posting, Through Window, Electric Meter Location, Foyer, Living Room, etc.).\n"
                        "2. OCCUPANCY ENFORCEMENT: If you detect a Lockbox, Vacant Sticker, servicer Posting (e.g. A2Z, Safeguard, Cyprexx), "
                        "overflowing uncollected mail, or a bare/empty interior, occupancy_status MUST BE 'Vacant'. A mowed lawn NEVER overrides this.\n"
                        "3. ACCESS: If interior photos were taken peering Through Window, interior_access_gained = False. "
                        "Mark interior_access_gained = True ONLY if physical entry inside the property was photographed."
                    )

                    candidate_models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
                    audit_success = False
                    last_err = ""

                    for mod in candidate_models:
                        for attempt in range(2):
                            try:
                                response = client.models.generate_content(
                                    model=mod,
                                    contents=[*images, prompt],
                                    config=types.GenerateContentConfig(
                                        response_mime_type="application/json",
                                        response_schema=InspectionAudit,
                                        temperature=0.0
                                    )
                                )
                                st.session_state["audit_data"] = json.loads(response.text)
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

# 4. Results Dashboard
if "audit_data" in st.session_state and "cached_images" in st.session_state:
    data = st.session_state["audit_data"]
    images = st.session_state["cached_images"]
    labels = data.get("photo_classifications", [])

    st.markdown("### Labeled Photo Evidence (InspectorADE Standards)")
    labeled_cols = st.columns(6)
    for idx, img in enumerate(images):
        with labeled_cols[idx % 6]:
            st.image(img, use_container_width=True)
            lbl = labels[idx] if idx < len(labels) else f"Photo {idx+1}"
            st.markdown(f'<div class="photo-tag" title="{lbl}">{lbl}</div>', unsafe_allow_html=True)

    st.divider()

    status_class = "badge-occupied" if data["occupancy_status"] == "Occupied" else "badge-vacant"
    st.markdown(f"### Audit Verdict: <span class='{status_class}'>{data['occupancy_status'].upper()}</span>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**Resident Type:** {data['occupied_by']}")
        st.write(f"**Determination Method:** {data['occupancy_determination_method']}")
        indicators = ", ".join(data["visual_indicators_found"]) if data["visual_indicators_found"] else "None detected"
        st.write(f"**Visual Indicators:** {indicators}")
        
    with c2:
        st.write(f"**Physical Interior Entry:** {'YES' if data['interior_access_gained'] else 'NO'}")
        st.write(f"**Interior Condition:** {data['interior_condition']}")
        st.write(f"**Interior Debris/Hazards:** {'YES' if data['interior_debris_present'] else 'None'}")
        
    with c3:
        st.write(f"**Stories / Build:** {data['property_stories']} Story | {data['construction_type']}")
        st.write(f"**Attached Garage:** {'Yes' if data['attached_garage_present'] else 'No'}")
        st.write(f"**Electric Meter:** {'Installed' if data.get('electric_meter_installed') else 'Missing/Unobserved'}")
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
