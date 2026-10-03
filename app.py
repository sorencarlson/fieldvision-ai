import concurrent.futures
import io
import json
import os
import time
from typing import List, Literal, Optional
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
import streamlit as st

st.set_page_config(
    page_title="FieldVision AI Turbo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "uploader_key" not in st.session_state:
  st.session_state["uploader_key"] = 0

# Optimized Visual Presentation
st.markdown(
    """
<style>
    .main-header { font-size: 1.8rem; font-weight: 800; text-align: center; margin-bottom: 0.1rem; color: #1e293b; }
    .sub-header { font-size: 0.88rem; color: #64748b; text-align: center; margin-bottom: 1.2rem; }
    .badge-occupied { background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 1rem; }
    .badge-vacant { background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 1rem; }
    .photo-tag { 
        background-color: #0f172a; 
        color: #f8fafc; 
        font-size: 0.65rem; 
        font-weight: 600; 
        text-align: center; 
        padding: 4px 5px; 
        border-radius: 4px; 
        margin-top: 3px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">FieldVision AI High-Throughput Engine</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Field-Speed Multi-Threaded Processor •'
    ' InspectorADE Standard</div>',
    unsafe_allow_html=True,
)

# InspectorADE Complete Taxonomy
PhotoLabel = Literal[
    "Street Sign",
    "Street Scene",
    "Front Yard",
    "House Number / Address",
    "Property to Left",
    "Property to Right",
    "Foundation",
    "Roof Condition",
    "Roof Damage",
    "Lockbox",
    "Missing Lockbox",
    "Vacant Sticker",
    "Posting",
    "Through Window",
    "Doors Need Securing",
    "Windows Boarded",
    "Windows Broken",
    "Unable to access interior",
    "No Trespassing",
    "Key Working",
    "Electric Meter Location",
    "Electric Meter Location Missing",
    "Water Meter",
    "Water Shutoff",
    "Water Tank",
    "Water Heater Location",
    "Water Heater Location Missing",
    "Open Breaker Box",
    "Furnace",
    "Sump Pump",
    "Propane Tank",
    "Oil Tank",
    "Volt Stick",
    "Foyer",
    "Living Room",
    "Living Room Condition",
    "Family Room",
    "Kitchen",
    "Kitchen Condition",
    "Master Bathroom",
    "Half Bathroom",
    "Utility Room",
    "Sun Porch",
    "Stairway Condition",
    "Hallway Condition",
    "Interior Debris",
    "Exterior Debris",
    "Interior Health Hazard",
    "Exterior Health Hazard",
    "Interior Personal Property",
    "Exterior Personal Property",
    "Vandalism",
    "Water Damage",
    "Freeze Damage",
    "Fire Damage",
    "Mold",
    "Mortgagor Neglect",
    "Wear and Tear",
    "Gutters/Downspouts Damaged",
    "Handrails Damages/Missing",
    "Holes/Trip Hazards",
    "Outbuilding",
    "Outbuilding Condition",
    "Garage",
    "Garage Condition",
    "Fence",
    "Pool Condition",
    "Pool fence/gate/lanai",
    "VIN#/Plate",
    "Yard Condition",
    "Other / Unclassified",
]


class BatchLabels(BaseModel):
  labels: List[PhotoLabel] = Field(
      description=(
          "Exact sequential list of InspectorADE labels matching every image in"
          " this batch."
      )
  )


class MasterAuditReport(BaseModel):
  occupancy_status: Literal["Occupied", "Vacant", "Unknown"]
  occupied_by: Literal[
      "Owner", "Tenant", "Vacant/None", "Vagrant/Squatter", "Unknown"
  ]
  occupancy_determination_method: Literal["Visual", "Direct contact", "Other"]
  visual_indicators_found: List[str]
  property_stories: Literal["1", "2", "3", "4", "5"]
  construction_type: Literal[
      "Brick/Block",
      "Frame",
      "Frame and brick",
      "Stone",
      "Stucco",
      "Vinyl frame",
      "Other",
  ]
  attached_garage_present: bool
  electric_meter_installed: bool
  interior_access_gained: bool
  interior_condition: Literal["Good", "Fair", "Poor", "Not Inspected"]
  interior_debris_present: bool
  exterior_condition: Literal["Good", "Fair", "Poor"]
  exterior_damage_present: bool
  exterior_damage_details: Optional[str] = "None"
  bank_narrative: str = Field(
      description=(
          "A crisp, 2-3 sentence factual loan servicer summary citing"
          " lockboxes, postings, interior state, and structural condition."
      )
  )


def compress_for_inference(uploaded_file, max_dim=1400, quality=78) -> Image.Image:
  """In-memory stream compression: keeps crisp sharpness for lockboxes/tags while slicing upload weight by 80%."""
  img = Image.open(uploaded_file)
  img = ImageOps.exif_transpose(img)
  if img.mode in ("RGBA", "P"):
    img = img.convert("RGB")
  img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)

  buf = io.BytesIO()
  img.save(buf, format="JPEG", quality=quality, optimize=True)
  buf.seek(0)
  return Image.open(buf)


def classify_batch_worker(
    client, batch_images, batch_idx, candidate_models
) -> List[str]:
  """Worker thread to label a batch in parallel."""
  prompt = (
      "Expert mortgage inspection photo classifier (InspectorADE standards).\n"
      "Assign the most accurate PhotoLabel to every image in this exact batch"
      " sequence.\n"
      "Distinguish carefully: Lockbox, Vacant Sticker, Posting, Through Window,"
      " Electric Meter Location, Living Room, Kitchen, Roof Condition,"
      " Foundation, etc."
  )
  for mod in candidate_models:
    try:
      resp = client.models.generate_content(
          model=mod,
          contents=[*batch_images, prompt],
          config=types.GenerateContentConfig(
              response_mime_type="application/json",
              response_schema=BatchLabels,
              temperature=0.0,
          ),
      )
      return json.loads(resp.text)["labels"]
    except Exception:
      time.sleep(0.5)
  return ["Other / Unclassified"] * len(batch_images)


# API Key
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if not api_key:
  with st.expander("API Configuration", expanded=True):
    api_key = st.text_input("Enter Gemini API Key", type="password")

uploaded_files = st.file_uploader(
    "Upload Inspection Photo Pack (Supports 75+ Photos)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}",
)

if uploaded_files:
  total_count = len(uploaded_files)

  if "cached_images" not in st.session_state:
    with st.spinner(f"Preparing {total_count} photos for high-speed audit..."):
      with concurrent.futures.ThreadPoolExecutor() as executor:
        st.session_state["cached_images"] = list(
            executor.map(compress_for_inference, uploaded_files)
        )

  images = st.session_state["cached_images"]

  if "audit_data" not in st.session_state:
    st.info(f"{total_count} inspection photos primed in field memory.")

    # High-density preview
    cols = st.columns(8)
    for idx, img in enumerate(images[:24]):
      with cols[idx % 8]:
        st.image(img, use_container_width=True)
    if total_count > 24:
      st.caption(f"...and {total_count - 24} more photos queued for audit.")

    if st.button(
        f"RUN LIGHTNING AUDIT ({total_count} PHOTOS)",
        type="primary",
        use_container_width=True,
    ):
      if not api_key:
        st.error("API Key required.")
      else:
        start_time = time.time()
        client = genai.Client(api_key=api_key)
        candidate_models = [
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ]

        # 1. Parallel Batching for 75 Photos (Chunks of 12-15 images)
        batch_size = 15
        batches = [
            images[i : i + batch_size]
            for i in range(0, len(images), batch_size)
        ]

        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text(
            f"Classifying {total_count} photos in parallel across"
            f" {len(batches)} threads..."
        )

        all_labels = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(batches)
        ) as executor:
          futures = [
              executor.submit(
                  classify_batch_worker, client, b, idx, candidate_models
              )
              for idx, b in enumerate(batches)
          ]
          for f in concurrent.futures.as_completed(futures):
            pass

          # Gather in order
          for fut in futures:
            all_labels.extend(fut.result())

        progress_bar.progress(70)
        progress_text.text("Synthesizing Master Mortgage Audit & Compliance Form...")

        # 2. Master Summary Inference
        # Synthesize audit data using representative property photos + extracted labels
        sample_indices = [
            0,
            1,
            2,
            min(5, total_count - 1),
            min(15, total_count - 1),
            total_count - 1,
        ]
        sample_photos = [images[i] for i in sorted(set(sample_indices))]

        summary_prompt = (
            "You are a master mortgage inspection audit reviewer.\n"
            f"Based on the classified labels for this {total_count}-photo pack: {all_labels}\n"
            "and the provided architectural anchor photos, compile the final bank inspection audit.\n"
            "RULES:\n"
            "1. If labels include 'Lockbox', 'Vacant Sticker', 'Posting', 'Through Window', or bare room conditions, "
            "occupancy_status MUST BE 'Vacant'. A cut lawn never overrides this.\n"
            "2. If interior rooms are present (Foyer, Living Room, Kitchen, etc.) and NOT just 'Through Window', interior_access_gained = True."
        )

        report_data = None
        for mod in candidate_models:
          try:
            res = client.models.generate_content(
                model=mod,
                contents=[*sample_photos, summary_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MasterAuditReport,
                    temperature=0.0,
                ),
            )
            report_data = json.loads(res.text)
            break
          except Exception:
            time.sleep(0.5)

        elapsed = round(time.time() - start_time, 1)
        progress_bar.progress(100)
        progress_text.empty()

        st.session_state["audit_data"] = report_data
        st.session_state["labels"] = all_labels
        st.session_state["elapsed"] = elapsed
        st.rerun()

# 4. Results Dashboard
if "audit_data" in st.session_state and "cached_images" in st.session_state:
  data = st.session_state["audit_data"]
  images = st.session_state["cached_images"]
  labels = st.session_state.get("labels", [])
  elapsed = st.session_state.get("elapsed", 0)

  st.success(
      f"Audit complete: {len(images)} photos labeled and report compiled in"
      f" {elapsed} seconds!"
  )

  status_class = (
      "badge-occupied"
      if data["occupancy_status"] == "Occupied"
      else "badge-vacant"
  )
  st.markdown(
      f"### Verdict: <span"
      f" class='{status_class}'>{data['occupancy_status'].upper()}</span>"
      f" &nbsp;|&nbsp; Resident: <b>{data['occupied_by']}</b>",
      unsafe_allow_html=True,
  )

  c1, c2, c3 = st.columns(3)
  with c1:
    st.write(f"**Determination Method:** {data['occupancy_determination_method']}")
    st.write(
        f"**Visual Indicators:** {', '.join(data['visual_indicators_found']) if data['visual_indicators_found'] else 'None'}"
    )

  with c2:
    st.write(
        f"**Physical Interior Entry:**"
        f" {'YES' if data['interior_access_gained'] else 'NO'}"
    )
    st.write(f"**Interior Condition:** {data['interior_condition']}")
    st.write(
        f"**Interior Debris/Hazards:**"
        f" {'YES' if data['interior_debris_present'] else 'None'}"
    )

  with c3:
    st.write(
        f"**Stories / Build:** {data['property_stories']} Story |"
        f" {data['construction_type']}"
    )
    st.write(
        f"**Attached Garage:**"
        f" {'Yes' if data['attached_garage_present'] else 'No'}"
    )
    st.write(
        f"**Electric Meter:**"
        f" {'Installed' if data.get('electric_meter_installed') else 'Missing/Unobserved'}"
    )

  st.info(f"**Bank-Ready Narrative:**\n\n{data['bank_narrative']}")

  # Gallery Display (8 Columns for 75+ Photos)
  st.markdown("### Labeled Photo Inventory (InspectorADE)")
  grid_cols = st.columns(8)
  for idx, img in enumerate(images):
    with grid_cols[idx % 8]:
      st.image(img, use_container_width=True)
      lbl = labels[idx] if idx < len(labels) else f"Photo {idx+1}"
      st.markdown(
          f'<div class="photo-tag" title="{lbl}">{lbl}</div>',
          unsafe_allow_html=True,
      )

  st.divider()
  if st.button("Clear / Next Inspection", use_container_width=True):
    st.session_state["uploader_key"] += 1
    st.session_state.pop("audit_data", None)
    st.session_state.pop("cached_images", None)
    st.session_state.pop("labels", None)
    st.rerun()
