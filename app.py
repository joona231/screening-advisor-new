
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageCms

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "training_data.csv"
ICC_PATH = APP_DIR / "FOGRA39L_coated.icc"
METRICS_PATH = APP_DIR / "validation_metrics.csv"
MEANS_PATH = APP_DIR / "experimental_means.csv"

PAPERS = [
    "ofsetni papir 80 g",
    "matirani papir 135 g",
    "ofsetni papir 170 g",
    "teksturirani papir 300 g",
]
PAPER_LABELS = {
    "ofsetni papir 80 g": "Offset 80 g",
    "matirani papir 135 g": "Matirani 135 g",
    "ofsetni papir 170 g": "Offset 170 g",
    "teksturirani papir 300 g": "Teksturirani 300 g",
}
SCREENINGS = ["150 dot","200 dot","200 line","300 dot","600 dot","stochastic"]

st.set_page_config(page_title="AI-PQI | Screening Advisor", layout="wide")

st.markdown("""
<style>
:root { --ink:#111827; --muted:#667085; --line:#e5e7eb; --soft:#f8fafc; --accent:#111827; }
html, body, [class*="css"] { font-family: Arial, "Segoe UI", sans-serif; }
[data-testid="stFileUploaderDropzone"] small, .stFileUploader small { display:none !important; }
.block-container { max-width:1120px; padding-top:2rem; padding-bottom:3rem; }
.hero { padding:0 0 1.25rem; }
.eyebrow { color:#667085; font-size:.72rem; letter-spacing:.13em; text-transform:uppercase; font-weight:700; }
.hero h1 { color:#111827; font-size:2.45rem; letter-spacing:-.055em; margin:.25rem 0 .35rem; }
.hero p { color:#667085; margin:0; font-size:.98rem; }
.condition { margin-top:.85rem; color:#98a2b3; font-size:.76rem; }
.card { border:1px solid var(--line); border-radius:18px; padding:1.2rem 1.25rem; background:#fff; }
.section { color:#667085; font-size:.72rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; margin-bottom:.75rem; }
.rec { border:1px solid #d9dee7; border-radius:20px; padding:1.45rem 1.55rem; background:linear-gradient(180deg,#fbfcfd,#f6f8fb); }
.rec-kicker { color:#667085; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; }
.rec-paper { color:#111827; font-size:1.42rem; font-weight:750; margin-top:.3rem; }
.rec-screen { color:#475467; font-size:1rem; margin-top:.05rem; }
.rec-number { color:#111827; font-size:2.65rem; font-weight:750; letter-spacing:-.055em; margin-top:.35rem; }
.note { color:#667085; font-size:.76rem; line-height:1.45; }
.metricbox { border:1px solid var(--line); border-radius:15px; padding:.9rem 1rem; background:#fff; min-height:92px; }
.metricbox .label { color:#667085; font-size:.72rem; }
.metricbox .value { color:#111827; font-size:1.25rem; font-weight:700; margin-top:.18rem; }
.badge { display:inline-block; border:1px solid #d0d5dd; border-radius:999px; padding:.25rem .55rem; color:#475467; font-size:.7rem; }
.warning { border-left:3px solid #98a2b3; padding:.65rem .8rem; background:#f8fafc; color:#475467; border-radius:0 10px 10px 0; font-size:.78rem; }
.footer { color:#98a2b3; font-size:.68rem; text-align:center; padding-top:2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    required = {"paper","screening","patch","C","M","Y","K","L_t","a_t","b_t","de00"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if len(df) != 9600:
        raise ValueError(f"Expected 9,600 measurements, found {len(df)}.")
    return df

@st.cache_data(show_spinner=False)
def load_metrics():
    return pd.read_csv(METRICS_PATH)

@st.cache_data(show_spinner=False)
def load_means():
    return pd.read_csv(MEANS_PATH)

@st.cache_resource(show_spinner=False, max_entries=1)
def icc_profiles():
    cmyk = ImageCms.getOpenProfile(str(ICC_PATH))
    lab = ImageCms.createProfile("LAB")
    cmyk_to_lab = ImageCms.buildTransformFromOpenProfiles(cmyk, lab, "CMYK", "LAB")
    srgb = ImageCms.createProfile("sRGB")
    srgb_to_cmyk = ImageCms.buildTransformFromOpenProfiles(srgb, cmyk, "RGB", "CMYK")
    return cmyk_to_lab, srgb_to_cmyk

def lab_from_cmyk_icc(cmyk):
    transform, _ = icc_profiles()
    x = np.asarray(cmyk, dtype=float).reshape(-1, 4)
    px = np.clip(np.rint(x * 255.0 / 100.0), 0, 255).astype(np.uint8)

    results = []
    for row in px:
        im = Image.new("CMYK", (1, 1), tuple(int(v) for v in row))
        out = ImageCms.applyTransform(im, transform)
        # Pillow's 8-bit LAB image stores:
        # L8 = L* * 255/100, A8 = a* + 128, B8 = b* + 128.
        L8, A8, B8 = out.getpixel((0, 0))
        results.append([
            float(L8) * 100.0 / 255.0,
            float(A8) - 128.0,
            float(B8) - 128.0,
        ])
    return np.asarray(results, dtype=float)

def rgb_to_cmyk_icc(image):
    _, transform = icc_profiles()
    if image.mode != "RGB":
        image = image.convert("RGB")
    return ImageCms.applyTransform(image, transform)

def knn_predict(train_X, train_y, query_X, k=12):
    train_X = np.asarray(train_X, float)
    train_y = np.asarray(train_y, float)
    query_X = np.asarray(query_X, float)
    scale = np.array([100.,100.,100.,100.])
    tx = train_X / scale
    qx = query_X / scale
    out = np.empty(len(qx), dtype=float)
    k = min(k, len(tx))
    for start in range(0, len(qx), 256):
        q = qx[start:start+256]
        d = np.sqrt(((q[:,None,:]-tx[None,:,:])**2).sum(axis=2))
        idx = np.argpartition(d, k-1, axis=1)[:,:k]
        kd = np.take_along_axis(d, idx, axis=1)
        yy = train_y[idx]
        w = 1/np.maximum(kd, 1e-8)
        out[start:start+len(q)] = (w*yy).sum(axis=1)/w.sum(axis=1)
    return np.maximum(out, 0)

@st.cache_data(show_spinner=False)
def format_lab_display(lab):
    """Standard CIELAB display: L*=0..100, a*/b* centered at 0."""
    L, a, b = [float(v) for v in lab]
    return L, a, b

def reference_lab(df, cmyk_tuple):
    cmyk = np.array(cmyk_tuple, float)
    exact = df[
        np.isclose(df.C,cmyk[0]) & np.isclose(df.M,cmyk[1]) &
        np.isclose(df.Y,cmyk[2]) & np.isclose(df.K,cmyk[3])
    ]
    if len(exact):
        r = exact.iloc[0]
        return np.array([r.L_t,r.a_t,r.b_t],float), "experimental reference patch"
    return lab_from_cmyk_icc(cmyk)[0], "FOGRA39 ICC reference"

@st.cache_data(show_spinner=False)
def predict_single(df, cmyk_tuple):
    cmyk = np.array(cmyk_tuple,float)
    rows=[]
    for paper in PAPERS:
        d=df[df.paper==paper]
        for s in SCREENINGS:
            tr=d[d.screening==s]
            p=knn_predict(tr[["C","M","Y","K"]].to_numpy(), tr.de00.to_numpy(), cmyk.reshape(1,4), 12)[0]
            rows.append([paper,s,float(p)])
    return pd.DataFrame(rows,columns=["paper","screening","pred_de00"]).sort_values("pred_de00").reset_index(drop=True)

def prepare_image_for_analysis(image, max_side=1600):
    """Downsample before colour conversion to keep Streamlit memory use bounded."""
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image

def image_samples(image, max_samples=700):
    # Never materialize the original full-resolution image as a NumPy array.
    image = prepare_image_for_analysis(image)
    arr = np.asarray(image.convert("CMYK"), dtype=np.uint8)
    c = np.rint(arr.astype(np.float32) / 255 * 25) / 25 * 100
    flat = c.reshape(-1, 4).astype(np.float32)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    if len(uniq) > max_samples:
        idx = np.argsort(counts)[::-1][:max_samples]
        uniq, counts = uniq[idx], counts[idx]
    return uniq, counts / counts.sum()

def predict_image(df, colors, weights):
    rows=[]
    for paper in PAPERS:
        d=df[df.paper==paper]
        for s in SCREENINGS:
            tr=d[d.screening==s]
            p=knn_predict(tr[["C","M","Y","K"]].to_numpy(),tr.de00.to_numpy(),colors,12)
            rows.append([paper,s,float(np.sum(p*weights)),float(np.quantile(p,.90))])
    return pd.DataFrame(rows,columns=["paper","screening","mean_de00","p90_de00"]).sort_values("mean_de00").reset_index(drop=True)

df=load_data()
metrics=load_metrics()
means=load_means()

st.markdown("## Screening Advisor")
st.markdown("**Artificial Intelligence-assisted Print Quality Intelligence (AI-PQI)**")
st.caption("Data-driven decision support for paper and screening selection.")
st.caption("CIELAB scale: L* = 0-100 | a* and b* are centered around 0")
st.caption("FOGRA39 | D50 | 10 degree observer | 4 substrates | 6 screening methods | 9,600 measurements")

tab1,tab2,tab3=st.tabs(["Single tone","Image","Research"])

with tab1:
    st.markdown('<div class="section">CMYK input</div>',unsafe_allow_html=True)
    cols=st.columns(4)
    vals=[]
    for col,ch in zip(cols,["C","M","Y","K"]):
        with col:
            vals.append(st.number_input(ch,0.0,100.0,20.0,1.0,key="v_"+ch))
    st.caption("CMYK range: 0-100 for each channel")
    cmyk=tuple(float(x) for x in vals)
    lab,lab_source=reference_lab(df,cmyk)
    exact_source="Experimental reference" if lab_source.startswith("experimental") else "FOGRA39 ICC"
    st.markdown('<div style="height:.8rem"></div>',unsafe_allow_html=True)
    st.markdown("**Target / reference CIELAB**")
    labcols=st.columns(3)
    for col,label,value in zip(labcols,["L*","a*","b*"],lab):
        with col:
            st.metric(label, f"{float(value):.2f}")
    st.caption("L* is lightness (0 = black, 100 = ideal reference white). a* and b* are centered around 0. The displayed values are CIELAB, not raw 8-bit Lab channels.")
    info1,info2=st.columns(2)
    with info1:
        st.markdown(f'<div class="metricbox"><div class="label">Reference source</div><div class="value">{exact_source}</div><div class="note">Exact chart CMYK uses the experimental Target Lab; other CMYK values use the FOGRA39 ICC reference transform.</div></div>',unsafe_allow_html=True)
    with info2:
        st.markdown('<div class="metricbox"><div class="label">Decision space</div><div class="value">24 conditions</div><div class="note">4 papers × 6 screening methods</div></div>',unsafe_allow_html=True)
    res=predict_single(df,cmyk)
    best=res.iloc[0]
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="rec"><div class="rec-kicker">AI-PQI recommendation</div><div class="rec-paper">{PAPER_LABELS[best.paper]}</div><div class="rec-screen">{best.screening}</div><div class="rec-number">{best.pred_de00:.2f} DeltaE00</div><div class="note">Predicted colour difference under the validated experimental conditions.</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Comparison</div>',unsafe_allow_html=True)
    show=res.copy()
    show["paper"]=show.paper.map(PAPER_LABELS)
    show.columns=["Paper","Screening","Predicted DeltaE00"]
    show["Predicted DeltaE00"]=show["Predicted DeltaE00"].round(2)
    st.dataframe(show,use_container_width=True,hide_index=True)

with tab2:
    st.markdown('<div class="section">Image analysis</div>',unsafe_allow_html=True)
    st.markdown("**Image upload**")
    if "image_uploader_version" not in st.session_state:
        st.session_state.image_uploader_version = 0
    uploaded=st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        label_visibility="visible",
        key=f"image_uploader_{st.session_state.image_uploader_version}"
    )
    mode=st.radio("Input colour space",["CMYK image","RGB image to FOGRA39 CMYK"],horizontal=True)
    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        allowed_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        if suffix not in allowed_suffixes:
            st.error("Unsupported format. Please upload a JPEG, PNG or TIFF image.")
            st.stop()
        img=Image.open(uploaded)
        img=prepare_image_for_analysis(img)
        if mode.startswith("RGB"):
            work=rgb_to_cmyk_icc(img)
            note="RGB input converted with sRGB to FOGRA39 ICC colour management."
        else:
            work=img.convert("CMYK")
            note="CMYK values analysed directly; no RGB conversion was applied."
        st.image(img,caption="Input preview",use_container_width=True)
        if st.button("Remove image", key=f"remove_image_{st.session_state.image_uploader_version}", type="tertiary"):
            st.session_state.image_uploader_version += 1
            st.rerun()
        st.markdown(f'<div class="warning">{note}</div>',unsafe_allow_html=True)
        colors,weights=image_samples(work)
        ir=predict_image(df,colors,weights)
        best=ir.iloc[0]
        st.markdown(f'<div class="warning"><b>Analysis principle.</b> The image is analysed as a distribution of CMYK colours. Each quantized CMYK colour is weighted by its pixel frequency, so the primary decision criterion is the <b>pixel-weighted mean predicted DeltaE00</b> across the image. The dominant colour alone is not used for the recommendation. P90 DeltaE00 is shown as a secondary measure of the higher-error part of the colour distribution. Current analysis uses {len(colors)} representative CMYK colours; the most frequent colour represents {float(np.max(weights))*100:.1f}% of analysed pixels.</div>',unsafe_allow_html=True)
        st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="rec"><div class="rec-kicker">AI-PQI image recommendation</div><div class="rec-paper">{PAPER_LABELS[best.paper]}</div><div class="rec-screen">{best.screening}</div><div class="rec-number">{best.mean_de00:.2f} DeltaE00</div><div class="note">Primary criterion: pixel-weighted mean predicted DeltaE00  |  P90: {best.p90_de00:.2f}</div></div>',unsafe_allow_html=True)
        show=ir.copy()
        show["paper"]=show.paper.map(PAPER_LABELS)
        show.columns=["Paper","Screening","Mean DeltaE00","P90 DeltaE00"]
        show[["Mean DeltaE00","P90 DeltaE00"]]=show[["Mean DeltaE00","P90 DeltaE00"]].round(2)
        st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
        st.dataframe(show,use_container_width=True,hide_index=True)

with tab3:
    st.markdown('<div class="section">AI-PQI validation</div>',unsafe_allow_html=True)
    st.caption("Five-fold grouped cross-validation by Patch ID. The same patch is never used simultaneously for training and testing.")
    cols=st.columns(4)
    overall_acc=np.average(metrics.recommendation_accuracy,weights=[400]*len(metrics))
    for col,label,value in zip(cols,["Recommendation accuracy","Mean MAE","Mean RMSE","Validation design"],
                               [f"{overall_acc*100:.1f}%","{:.2f} DeltaE00".format(metrics.mae.mean()),"{:.2f} DeltaE00".format(metrics.rmse.mean()),"5-fold grouped"]):
        with col:
            st.markdown(f'<div class="metricbox"><div class="label">{label}</div><div class="value">{value}</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Validation by substrate</div>',unsafe_allow_html=True)
    v=metrics.copy()
    v["paper"]=v.paper.map(PAPER_LABELS)
    v["recommendation_accuracy"]=(v.recommendation_accuracy*100).round(1)
    v["mae"]=v.mae.round(2)
    v["rmse"]=v.rmse.round(2)
    v["r2"]=v.r2.round(2)
    v.columns=["Paper","Recommendation accuracy %","MAE DeltaE00","RMSE DeltaE00","R²"]
    st.dataframe(v,use_container_width=True,hide_index=True)
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Data provenance</div>',unsafe_allow_html=True)
    provenance = pd.DataFrame([
        ["Experimental measurements","4 substrates × 6 screenings × 400 patches","9,600 measurements","X-Rite i1Pro 2 / Xerox Colour C60/C70"],
        ["Reference chart","400 CMYK patches with experimental Target L*a*b*","400 reference colours","D50 / 10° observer"],
        ["Model input","CMYK values from the measured patches","4,800 training conditions per?","Weighted 12-nearest-neighbour regression"],
        ["Colour management","FOGRA39 ICC","Reference conversion for arbitrary CMYK and RGB→CMYK","Not a replacement for measured Target Lab"],
    ], columns=["Source / component","What it contains","Scope","Role in AI-PQI"])
    provenance.iloc[2,2] = "4 papers × 6 screenings × 400 patches"
    st.dataframe(provenance,use_container_width=True,hide_index=True)
    st.caption("The Research panel documents the experimental source of the model and separates measured data from the FOGRA39 colour-management reference. It is not intended to fetch external literature automatically.")
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Experimental mean DeltaE00</div>',unsafe_allow_html=True)
    mm=means.pivot(index="paper",columns="screening",values="mean_measured_de00").loc[PAPERS,SCREENINGS]
    mm.index=[PAPER_LABELS[x] for x in mm.index]
    st.dataframe(mm.round(2),use_container_width=True)
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    st.markdown('<div class="warning"><b>QC note.</b> The 80 g offset / 150 dot condition contains a systematic group of unusually high DeltaE00 values. These observations are retained in the dataset rather than silently removed; the effect should be discussed as a data-quality finding in the research paper.</div>',unsafe_allow_html=True)
    st.markdown('<div style="height:.8rem"></div>',unsafe_allow_html=True)
    st.caption("Model: weighted 12-nearest-neighbour regression in normalized CMYK space. Target: measured DeltaE00. FOGRA39 is used for colour-managed reference conversion; the experimental Target Lab values remain the reference data for the measured chart.")

st.markdown('<div class="footer">AI-PQI  |  Experimental decision-support prototype  |  FOGRA39  |  D50  |  10°</div>',unsafe_allow_html=True)
