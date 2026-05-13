import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, densenet121
from PIL import Image
import json
from io import BytesIO

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Cure AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-dark:     #1a3a2a;
    --green-mid:      #2d6a4f;
    --green-light:    #52b788;
    --green-pale:     #b7e4c7;
    --cream:          #f8f4ed;
    --amber:          #e9a84c;
    --red-soft:       #e05c5c;
    --shadow:         0 8px 32px rgba(26,58,42,0.18);
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2318 0%, #1a3a2a 40%, #0d2b1d 100%) !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 15% 10%, rgba(82,183,136,0.09) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 85% 85%, rgba(45,106,79,0.13) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}
[data-testid="stHeader"] { background: transparent !important; }

.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(82,183,136,0.15);
    border: 1px solid rgba(82,183,136,0.35);
    color: var(--green-light);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.35rem 1.1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 900;
    line-height: 1.05;
    color: var(--cream);
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}
.hero-title span { color: var(--green-light); }
.hero-sub {
    font-size: 1.05rem;
    color: rgba(248,244,237,0.5);
    font-weight: 300;
    max-width: 500px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.card-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--cream);
    margin-bottom: 1rem;
}

/* ── Result banner ── */
.result-healthy {
    background: linear-gradient(135deg, rgba(82,183,136,0.2), rgba(45,106,79,0.3));
    border: 1px solid rgba(82,183,136,0.5);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.5s ease;
}
.result-disease {
    background: linear-gradient(135deg, rgba(224,92,92,0.15), rgba(180,60,60,0.25));
    border: 1px solid rgba(224,92,92,0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.5s ease;
}
.result-plant {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--cream);
    margin: 0 0 0.15rem;
}
.result-cond {
    font-size: 1rem;
    color: rgba(248,244,237,0.65);
    margin: 0 0 1rem;
}
.confidence-big {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    color: var(--green-light);
    line-height: 1;
}
.meta-row {
    display: flex;
    gap: 1.5rem;
    align-items: flex-end;
    margin-top: 0.8rem;
    flex-wrap: wrap;
}
.meta-item-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(248,244,237,0.38);
    margin-bottom: 0.2rem;
}

/* ── Confidence bars ── */
.conf-item { margin-bottom: 0.75rem; }
.conf-label {
    font-size: 0.82rem;
    color: rgba(248,244,237,0.6);
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.25rem;
}
.conf-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    height: 7px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #2d6a4f, #52b788);
}
.conf-bar-fill.top {
    background: linear-gradient(90deg, #2d6a4f, #52b788, #95d5b2);
}

/* ── Treatment ── */
.treatment-wrap {
    background: linear-gradient(135deg, rgba(233,168,76,0.07), rgba(233,168,76,0.02));
    border: 1px solid rgba(233,168,76,0.22);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    animation: fadeUp 0.6s ease 0.15s both;
}
.treatment-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--amber);
    margin-bottom: 1.4rem;
}
.tgrid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}
.tblock {
    background: rgba(255,255,255,0.04);
    border-radius: 13px;
    padding: 1rem 1.2rem;
}
.tblock.organic   { border-left: 3px solid #95d5b2; }
.tblock.chemical  { border-left: 3px solid var(--amber); }
.tblock.prevention{ border-left: 3px solid #74c0fc; }
.tblock.note      { border-left: 3px solid var(--red-soft); }
.tblock-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(248,244,237,0.38);
    margin-bottom: 0.5rem;
}
.tblock-content {
    font-size: 0.875rem;
    color: rgba(248,244,237,0.82);
    line-height: 1.65;
}

/* ── Severity pills ── */
.sev-pill {
    display: inline-block;
    padding: 0.22rem 0.85rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
}
.sev-none   { background: rgba(82,183,136,0.18);  color: #95d5b2; }
.sev-medium { background: rgba(233,168,76,0.18);  color: var(--amber); }
.sev-high   { background: rgba(224,92,92,0.18);   color: #ff9090; }

/* ── Stat pills ── */
.stat-row { display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.8rem; }
.stat-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 100px;
    padding: 0.28rem 0.85rem;
    font-size: 0.76rem;
    color: rgba(248,244,237,0.55);
}
.stat-pill span { color: var(--green-light); font-weight: 600; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 0.76rem;
    color: rgba(248,244,237,0.22);
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 3rem;
    line-height: 2;
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(82,183,136,0.38) !important;
    border-radius: 16px !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(82,183,136,0.75) !important;
    background: rgba(82,183,136,0.06) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: rgba(248,244,237,0.5) !important; }
.stButton > button {
    background: linear-gradient(135deg, #2d6a4f, #52b788) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(82,183,136,0.3) !important;
}
div[data-testid="stImage"] img {
    border-radius: 14px !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
}
p, li, label { color: rgba(248,244,237,0.75) !important; }
</style>
""", unsafe_allow_html=True)

# ── Treatment Database (all 38 classes) ───────────────────────────────────
TREATMENTS = {
    "Apple___Apple_scab": {
        "severity": "medium",
        "organic":    "Apply sulfur-based fungicide or neem oil spray every 7–10 days during wet seasons. Remove and destroy all fallen leaves.",
        "chemical":   "Use Captan (50 WP) or Mancozeb at first sign of infection. Apply Myclobutanil for severe cases.",
        "prevention": "Plant resistant varieties. Prune for air circulation. Avoid overhead irrigation.",
        "note":       "Scab thrives in cool, wet weather (55–75°F). Early spring protection is critical."
    },
    "Apple___Black_rot": {
        "severity": "high",
        "organic":    "Copper-based fungicide (copper hydroxide). Prune infected branches 8–10 inches below visible damage.",
        "chemical":   "Captan or Thiophanate-methyl at 10-day intervals.",
        "prevention": "Remove mummified fruits. Sanitize pruning tools. Maintain orchard hygiene.",
        "note":       "Infected wood must be removed completely or the disease will keep spreading."
    },
    "Apple___Cedar_apple_rust": {
        "severity": "medium",
        "organic":    "Neem oil or sulfur spray applied from bud break through early summer.",
        "chemical":   "Myclobutanil (Rally) or Propiconazole at pink bud stage, repeated every 7–10 days.",
        "prevention": "Remove nearby juniper/cedar trees (alternate host). Plant rust-resistant apple varieties.",
        "note":       "This disease requires BOTH apple and juniper trees to complete its life cycle."
    },
    "Apple___healthy": {
        "severity": "none",
        "organic":    "Continue regular balanced fertilization with compost or organic matter.",
        "chemical":   "Maintain preventive spray schedule with copper or sulfur as needed.",
        "prevention": "Monitor weekly for early signs of disease. Maintain proper pruning and spacing.",
        "note":       "Your plant is healthy! Keep up the current care routine."
    },
    "Blueberry___healthy": {
        "severity": "none",
        "organic":    "Apply acidic mulch (pine bark/needles) to maintain soil pH 4.5–5.5.",
        "chemical":   "Use sulfur-based fertilizer to maintain soil acidity if needed.",
        "prevention": "Ensure well-draining acidic soil. Monitor for stem blight and mummy berry.",
        "note":       "Blueberries thrive with consistent acidity — test soil pH annually."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "medium",
        "organic":    "Spray potassium bicarbonate or diluted neem oil. Baking soda solution (1 tbsp/gallon) also works.",
        "chemical":   "Myclobutanil or Trifloxystrobin fungicide at 14-day intervals.",
        "prevention": "Avoid excess nitrogen fertilization. Ensure good air circulation through pruning.",
        "note":       "Powdery mildew spreads in warm days with cool nights and moderate humidity."
    },
    "Cherry_(including_sour)___healthy": {
        "severity": "none",
        "organic":    "Apply compost tea as a foliar spray to boost natural disease resistance.",
        "chemical":   "Preventive copper spray before bud break each spring.",
        "prevention": "Prune annually for airflow. Remove fallen fruit and leaves promptly.",
        "note":       "Cherry trees are healthy — watch for brown rot during fruiting season."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "high",
        "organic":    "Crop rotation with non-host plants for 1–2 seasons. Destroy infected crop residue after harvest.",
        "chemical":   "Azoxystrobin + Propiconazole (Quilt Xcel) or Pyraclostrobin at VT/R1 growth stage.",
        "prevention": "Plant resistant hybrids. Avoid continuous corn planting. Till residue after harvest.",
        "note":       "Gray leaf spot is worst in warm, humid conditions with poor air circulation."
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "medium",
        "organic":    "Remove heavily infected leaves. Crop rotation reduces inoculum levels.",
        "chemical":   "Mancozeb or Chlorothalonil at first sign. Propiconazole for severe infection.",
        "prevention": "Plant rust-resistant corn hybrids. Avoid late planting in rust-prone areas.",
        "note":       "Common rust spores are wind-carried — resistant varieties are the best long-term defense."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "high",
        "organic":    "Rotate crops and incorporate residue. Use Trichoderma-based biocontrol agents.",
        "chemical":   "Azoxystrobin, Propiconazole, or Tebuconazole fungicide at tasseling stage.",
        "prevention": "Use resistant hybrids (Ht1/Ht2 genes). Avoid dense planting. Till infected residue.",
        "note":       "Most damaging when infection occurs before or during tasseling."
    },
    "Corn_(maize)___healthy": {
        "severity": "none",
        "organic":    "Side-dress with compost nitrogen at V6 stage for continued plant health.",
        "chemical":   "Apply preventive fungicide at VT stage if disease pressure is historically high in your area.",
        "prevention": "Scout weekly for lesions. Maintain proper plant spacing (28,000–32,000 plants/acre).",
        "note":       "Corn looks great! Monitor for tar spot if you are in a high-risk region."
    },
    "Grape___Black_rot": {
        "severity": "high",
        "organic":    "Copper fungicide (Bordeaux mixture) from bud break. Remove all mummified berries immediately.",
        "chemical":   "Mancozeb or Myclobutanil every 7–10 days from budbreak through veraison.",
        "prevention": "Prune for open canopy. Remove infected fruit promptly. Train vines for airflow.",
        "note":       "Black rot can destroy 100% of the crop if not managed early in the season."
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "high",
        "organic":    "No fully effective organic treatment. Prune infected wood immediately after symptom appearance.",
        "chemical":   "No fully registered chemical control available. Focus on wound protection.",
        "prevention": "Protect all pruning wounds with fungicidal paste. Use sterilized pruning tools.",
        "note":       "Esca is a chronic trunk disease — prevention through wound protection is critical."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "medium",
        "organic":    "Neem oil or copper-based sprays. Improve canopy management for better airflow.",
        "chemical":   "Mancozeb or Captan fungicide applied every 10–14 days during growing season.",
        "prevention": "Avoid overhead irrigation. Remove and destroy fallen infected leaves after harvest.",
        "note":       "Late-season disease — rarely causes severe economic damage but reduces vine vigor."
    },
    "Grape___healthy": {
        "severity": "none",
        "organic":    "Apply compost or well-aged manure in spring. Foliar kelp spray boosts immunity.",
        "chemical":   "Maintain preventive Mancozeb schedule during wet periods.",
        "prevention": "Train canopy for sun exposure. Scout for downy mildew and botrytis regularly.",
        "note":       "Healthy vines! Ensure balanced nutrition — avoid excess nitrogen."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "high",
        "organic":    "No cure exists. Remove and destroy infected trees immediately to prevent grove-wide spread.",
        "chemical":   "Trunk injection of oxytetracycline shows partial results in research settings only.",
        "prevention": "Control Asian citrus psyllid vector with Imidacloprid. Use certified disease-free nursery stock.",
        "note":       "⚠️ HLB is INCURABLE and FATAL to the tree. Infected trees must be removed immediately."
    },
    "Peach___Bacterial_spot": {
        "severity": "medium",
        "organic":    "Copper hydroxide spray during dormant season. Kaolin clay as a physical barrier during season.",
        "chemical":   "Oxytetracycline (Mycoshield) or copper bactericides from shuck split through harvest.",
        "prevention": "Plant resistant varieties (Contender, Redhaven). Avoid overhead irrigation. Install windbreaks.",
        "note":       "Bacterial spot is worst after hail events — inspect trees immediately after any storm."
    },
    "Peach___healthy": {
        "severity": "none",
        "organic":    "Apply compost in early spring. Foliar fish emulsion boosts tree health.",
        "chemical":   "Dormant copper spray annually to prevent bacterial spot and brown rot.",
        "prevention": "Thin fruit to 6–8 inches apart. Prune for open center canopy.",
        "note":       "Peach tree is healthy! Watch for peach leaf curl in early spring."
    },
    "Pepper,_bell___Bacterial_spot": {
        "severity": "medium",
        "organic":    "Copper-based sprays (copper sulfate) every 5–7 days during wet weather.",
        "chemical":   "Copper hydroxide + Mancozeb tank mix. Acibenzolar-S-methyl (Actigard) as resistance inducer.",
        "prevention": "Use certified disease-free seed. Avoid working in wet fields. Rotate crops 2–3 years.",
        "note":       "Disease spreads rapidly in warm, rainy conditions — early detection is key."
    },
    "Pepper,_bell___healthy": {
        "severity": "none",
        "organic":    "Side-dress with compost at transplanting and fruit set stages.",
        "chemical":   "Apply preventive copper spray during prolonged wet periods.",
        "prevention": "Stake plants for air circulation. Mulch soil to reduce splash.",
        "note":       "Peppers are healthy! Monitor for aphids and thrips which can spread viruses."
    },
    "Potato___Early_blight": {
        "severity": "medium",
        "organic":    "Copper-based fungicide or neem oil every 7 days during humid conditions.",
        "chemical":   "Chlorothalonil, Mancozeb, or Azoxystrobin fungicide at 7–10 day intervals.",
        "prevention": "Rotate crops 3–4 years. Use certified seed. Destroy volunteer potato plants.",
        "note":       "Early blight worsens as plants age and under nutrient stress — maintain proper fertilization."
    },
    "Potato___Late_blight": {
        "severity": "high",
        "organic":    "Copper fungicides (Bordeaux mixture) offer partial protection. Remove all infected tissue immediately.",
        "chemical":   "Metalaxyl (Ridomil Gold), Cymoxanil, or Dimethomorph. Apply preventively before symptoms appear.",
        "prevention": "Use certified blight-free seed. Destroy cull piles. Hill soil to reduce tuber infection.",
        "note":       "⚠️ Late blight can destroy an entire field in under a week. Act at the FIRST sign."
    },
    "Potato___healthy": {
        "severity": "none",
        "organic":    "Hill soil around plants when 6 inches tall. Apply balanced compost.",
        "chemical":   "Apply preventive fungicide during prolonged wet periods (>12 hours leaf wetness).",
        "prevention": "Monitor daily during cool, wet weather. Scout for Colorado beetle.",
        "note":       "Potatoes are healthy! Keep soil consistently moist but never waterlogged."
    },
    "Raspberry___healthy": {
        "severity": "none",
        "organic":    "Mulch with straw to retain moisture and suppress weeds.",
        "chemical":   "Apply lime sulfur dormant spray to prevent cane diseases.",
        "prevention": "Remove old fruiting canes after harvest. Ensure plants are trellised and well-spaced.",
        "note":       "Raspberry canes are healthy! Watch for cane borer and spur blight."
    },
    "Soybean___healthy": {
        "severity": "none",
        "organic":    "Inoculate seed with Bradyrhizobium japonicum for optimal nitrogen fixation.",
        "chemical":   "Apply preventive pyraclostrobin at R3 if sudden death syndrome is common in your area.",
        "prevention": "Rotate with non-legumes every 2 years. Scout for sudden death syndrome symptoms.",
        "note":       "Soybeans look healthy! Monitor for Asian soybean rust if in southern growing regions."
    },
    "Squash___Powdery_mildew": {
        "severity": "medium",
        "organic":    "Potassium bicarbonate spray or diluted neem oil weekly. Milk solution (40% milk/60% water) is effective.",
        "chemical":   "Trifloxystrobin or Myclobutanil fungicide at first sign of white powdery spots.",
        "prevention": "Plant resistant varieties. Space plants widely. Avoid excess nitrogen fertilization.",
        "note":       "Powdery mildew doesn't need wet leaves to spread — warm days with cool nights favor it."
    },
    "Strawberry___Leaf_scorch": {
        "severity": "medium",
        "organic":    "Remove and destroy infected leaves. Apply compost tea to boost plant resistance.",
        "chemical":   "Myclobutanil or Captan fungicide every 7–10 days during wet spring weather.",
        "prevention": "Plant certified disease-free stock. Remove old foliage after bed renovation.",
        "note":       "Leaf scorch reduces photosynthesis and weakens plants — treat early to avoid yield loss."
    },
    "Strawberry___healthy": {
        "severity": "none",
        "organic":    "Apply balanced compost in early spring. Use straw mulch to prevent soil splash.",
        "chemical":   "Preventive Captan spray during bloom to prevent gray mold (Botrytis).",
        "prevention": "Remove runners to maintain plant vigor. Renovate beds annually after harvest.",
        "note":       "Strawberries are healthy! Watch for spider mites in hot, dry weather."
    },
    "Tomato___Bacterial_spot": {
        "severity": "medium",
        "organic":    "Copper bactericide spray every 5–7 days. Remove infected leaves promptly.",
        "chemical":   "Copper hydroxide + Mancozeb tank mix or Actigard (systemic resistance inducer).",
        "prevention": "Use disease-free certified seed. Avoid overhead watering. Stake all plants.",
        "note":       "Spreads through rain splash and contaminated tools — sanitize between plants."
    },
    "Tomato___Early_blight": {
        "severity": "medium",
        "organic":    "Copper or neem oil spray every 7 days. Remove lower leaves touching soil.",
        "chemical":   "Chlorothalonil, Mancozeb, or Azoxystrobin every 7–10 days.",
        "prevention": "Mulch soil to prevent splash. Stake plants. Rotate crops every 3 years.",
        "note":       "Starts on oldest/lowest leaves — remove infected leaves immediately to slow spread."
    },
    "Tomato___Late_blight": {
        "severity": "high",
        "organic":    "Copper fungicide at 5–7 day intervals. Remove and bag all infected tissue immediately.",
        "chemical":   "Metalaxyl (Ridomil), Cymoxanil, or Fluopicolide. Apply preventively in cool/wet weather.",
        "prevention": "Avoid overhead irrigation. Destroy infected plants at season end. Stake for airflow.",
        "note":       "⚠️ Same pathogen as the Irish Potato Famine. Can destroy an entire crop in 7–10 days."
    },
    "Tomato___Leaf_Mold": {
        "severity": "medium",
        "organic":    "Reduce humidity below 85%. Neem oil or copper spray especially in greenhouse settings.",
        "chemical":   "Chlorothalonil or Mancozeb. Difenoconazole for severe cases.",
        "prevention": "Ensure greenhouse ventilation. Space plants. Avoid wetting leaves when watering.",
        "note":       "Mainly a greenhouse disease — outdoor infections are rare and usually less severe."
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "medium",
        "organic":    "Remove infected leaves immediately. Copper fungicide or neem oil spray.",
        "chemical":   "Chlorothalonil, Mancozeb, or Azoxystrobin at 7–10 day intervals.",
        "prevention": "Mulch soil surface. Stake plants. Avoid working in wet fields.",
        "note":       "Spreads upward from bottom leaves — remove lower foliage as a preventive measure."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "medium",
        "organic":    "Spray plants forcefully with water to dislodge mites. Apply neem oil or insecticidal soap.",
        "chemical":   "Abamectin, Spiromesifen, or Bifenazate miticide. Rotate chemistries to prevent resistance.",
        "prevention": "Maintain adequate irrigation (drought stress worsens mite outbreaks). Encourage predatory mites.",
        "note":       "⚠️ This is a PEST, not a fungal disease — fungicides will not help. Use miticides only."
    },
    "Tomato___Target_Spot": {
        "severity": "medium",
        "organic":    "Remove infected leaves. Copper-based fungicide every 7–10 days.",
        "chemical":   "Pyraclostrobin, Azoxystrobin, or Boscalid + Pyraclostrobin (Pristine).",
        "prevention": "Stake plants. Avoid overhead irrigation. Remove crop debris after harvest.",
        "note":       "Often confused with early blight — look for concentric rings with a distinct yellow halo."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "high",
        "organic":    "No cure. Remove and destroy infected plants immediately. Use reflective mulch to repel whiteflies.",
        "chemical":   "Control whitefly vector with Imidacloprid or Spirotetramat insecticide.",
        "prevention": "Use TYLCV-resistant varieties. Install insect-proof netting. Monitor for whiteflies.",
        "note":       "⚠️ TYLCV is INCURABLE and spreads via whiteflies. Infected plants MUST be removed."
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "high",
        "organic":    "No cure. Remove infected plants. Wash hands and tools with soap between plants.",
        "chemical":   "Control aphid vector with neem oil or pyrethrin-based sprays.",
        "prevention": "Use certified virus-free seed. Control aphids strictly. Do not smoke near tomatoes.",
        "note":       "⚠️ Incurable viral disease. Prevention and strict vector control are the only options."
    },
    "Tomato___healthy": {
        "severity": "none",
        "organic":    "Apply compost or worm castings monthly. Foliar calcium spray prevents blossom end rot.",
        "chemical":   "Preventive copper spray during wet periods. Apply systemic fungicide at first fruit set.",
        "prevention": "Stake and prune suckers. Mulch to retain moisture and prevent soil splash.",
        "note":       "Tomatoes are healthy! Watch for blossom end rot if watering becomes irregular."
    },
}

# ── Constants ──────────────────────────────────────────────────────────────
PTH_FILE          = "best_hybrid_resnet_densenet (3).pth"
CLASSES_FILE      = "classes (1).json"
RESNET_FEAT_DIM   = 2048
DENSENET_FEAT_DIM = 1024
HIDDEN_DIM        = 1024
DROPOUT           = 0.6

# ── Model Definition ───────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x)

class HybridResNetDenseNet(nn.Module):
    def __init__(self, num_classes, hidden_dim=1024, dropout=0.6):
        super().__init__()
        res_model = resnet50(weights=None)
        self.resnet_backbone = nn.Sequential(*list(res_model.children())[:-1])
        den_model = densenet121(weights=None)
        self.densenet_backbone = den_model.features
        self.densenet_pool = nn.AdaptiveAvgPool2d((1, 1))
        fused_dim = RESNET_FEAT_DIM + DENSENET_FEAT_DIM
        self.fusion_norm = nn.BatchNorm1d(fused_dim)
        self.se_block = SEBlock(fused_dim, reduction=16)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x):
        res_feat = self.resnet_backbone(x).flatten(1)
        den_feat = F.relu(self.densenet_backbone(x), inplace=True)
        den_feat = self.densenet_pool(den_feat).flatten(1)
        fused = torch.cat([res_feat, den_feat], dim=1)
        fused = self.fusion_norm(fused)
        fused = self.se_block(fused)
        return self.classifier(fused)

# ── Load model (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(CLASSES_FILE) as f:
        class_names = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridResNetDenseNet(num_classes=len(class_names), hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    state_dict = torch.load(PTH_FILE, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, device

def preprocess(image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(device)

def predict(image, model, class_names, device, top_k=5):
    tensor = preprocess(image, device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_idx = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs, top_idx):
        raw   = class_names[idx.item()]
        plant = raw.split("___")[0].replace("_", " ")
        cond  = raw.split("___")[1].replace("_", " ")
        results.append({
            "raw": raw, "plant": plant, "condition": cond,
            "confidence": round(prob.item() * 100, 2),
            "is_healthy": "healthy" in raw.lower()
        })
    return results

# ════════════════════════════════════════════════════════════════════════════
# RENDER
# ════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤖 Hybrid ResNet50 + DenseNet121</div>
    <h1 class="hero-title">Plant<span>Cure</span> AI</h1>
    <p class="hero-sub">Upload a leaf photo for instant AI disease diagnosis with expert treatment recommendations</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("🌿 Loading AI model..."):
    try:
        model, class_names, device = load_model()
        st.success(f"✅ Model ready — {len(class_names)} plant disease classes loaded")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# Supported plants
with st.expander("🌱 Supported Plants (14 crops, 38 disease classes)"):
    plants = sorted(set(c.split("___")[0].replace("_", " ") for c in class_names))
    cols = st.columns(5)
    icons = ["🍎","🫐","🍒","🌽","🍇","🍊","🍑","🫑","🥔","🍓","🫘","🥒","🍓","🍅"]
    for i, p in enumerate(plants):
        cols[i % 5].markdown(f"{icons[i % len(icons)]} **{p}**")

st.markdown("---")

# Upload
uploaded_file = st.file_uploader(
    "📤 Upload a leaf image (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="For best results: close-up of a single leaf, plain background, good lighting"
)

if uploaded_file:
    image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

    col_img, col_res = st.columns([1, 1.4], gap="large")

    # Image panel
    with col_img:
        st.markdown('<div class="card"><div class="card-header">📷 Uploaded Leaf</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">Width <span>{image.size[0]}px</span></div>
            <div class="stat-pill">Height <span>{image.size[1]}px</span></div>
            <div class="stat-pill">Mode <span>{image.mode}</span></div>
        </div></div>""", unsafe_allow_html=True)

    # Results panel
    with col_res:
        with st.spinner("🔬 Analysing leaf..."):
            results = predict(image, model, class_names, device, top_k=5)
        top = results[0]

        treatment   = TREATMENTS.get(top["raw"])
        sev         = treatment["severity"] if treatment else "medium"
        sev_html    = (
            '<span class="sev-pill sev-none">✅ Healthy</span>'   if sev == "none"   else
            '<span class="sev-pill sev-medium">🟡 Moderate</span>' if sev == "medium" else
            '<span class="sev-pill sev-high">🔴 High Risk</span>'
        )
        banner_cls  = "result-healthy" if top["is_healthy"] else "result-disease"

        st.markdown(f"""
        <div class="{banner_cls}">
            <p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.13em;
                      color:rgba(248,244,237,0.4);margin:0 0 0.3rem">
                {'✅' if top['is_healthy'] else '⚠️'} Detection Result
            </p>
            <p class="result-plant">{top['plant']}</p>
            <p class="result-cond">{top['condition']}</p>
            <div class="meta-row">
                <div>
                    <div class="meta-item-label">Confidence</div>
                    <div class="confidence-big">{top['confidence']}%</div>
                </div>
                <div>
                    <div class="meta-item-label">Severity</div>
                    {sev_html}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top-5 bars
        st.markdown('<div class="card"><div class="card-header">📊 Top 5 Predictions</div>', unsafe_allow_html=True)
        for i, r in enumerate(results):
            cond_label = r["condition"][:30] + "…" if len(r["condition"]) > 30 else r["condition"]
            bar_cls    = "top" if i == 0 else ""
            rank_icon  = "🥇" if i == 0 else f"{i+1}."
            color      = "#52b788" if i == 0 else "rgba(248,244,237,0.45)"
            weight     = "700" if i == 0 else "400"
            st.markdown(f"""
            <div class="conf-item">
                <div class="conf-label">
                    <span>{rank_icon} {r['plant']} — {cond_label}</span>
                    <span style="color:{color};font-weight:{weight}">{r['confidence']}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill {bar_cls}" style="width:{min(r['confidence'],100)}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Treatment Recommendations ──────────────────────────────────────────
    st.markdown("---")

    if treatment:
        t_icon  = "🌿" if top["is_healthy"] else "💊"
        t_title = (
            f"Care Tips for Healthy {top['plant']}"
            if top["is_healthy"] else
            f"Treatment Plan — {top['plant']}: {top['condition']}"
        )
        st.markdown(f"""
        <div class="treatment-wrap">
            <div class="treatment-title">{t_icon} {t_title}</div>
            <div class="tgrid">
                <div class="tblock organic">
                    <div class="tblock-label">🌱 Organic Treatment</div>
                    <div class="tblock-content">{treatment['organic']}</div>
                </div>
                <div class="tblock chemical">
                    <div class="tblock-label">🧪 Chemical Treatment</div>
                    <div class="tblock-content">{treatment['chemical']}</div>
                </div>
                <div class="tblock prevention">
                    <div class="tblock-label">🛡️ Prevention</div>
                    <div class="tblock-content">{treatment['prevention']}</div>
                </div>
                <div class="tblock note">
                    <div class="tblock-label">📌 Expert Note</div>
                    <div class="tblock-content">{treatment['note']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No treatment data available for this class.")

    if top["confidence"] < 30:
        st.warning("⚠️ Low confidence. Try uploading a clearer close-up of a single leaf on a plain background.")

# Footer
st.markdown("""
<div class="footer">
    🌿 PlantCure AI &nbsp;·&nbsp; Hybrid ResNet50 + DenseNet121 &nbsp;·&nbsp; 38 Classes &nbsp;·&nbsp; PlantVillage Dataset<br>
    <span style="opacity:0.6">For informational purposes only. Consult an agricultural expert for critical crop decisions.</span>
</div>
""", unsafe_allow_html=True)