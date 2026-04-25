import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, f1_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
import pickle, io, os

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS · Churn AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CYBERPUNK CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

*, html, body { box-sizing: border-box; }

.stApp {
    background: #02040a;
    color: #b8c8e0;
    background-image:
        linear-gradient(rgba(0,255,200,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,0.02) 1px, transparent 1px);
    background-size: 40px 40px;
}
.stApp > header { background: transparent !important; }

section[data-testid="stSidebar"] {
    background: #050a14 !important;
    border-right: 1px solid #0a3d2e;
}

/* ── Glitch header ── */
.nexus-header {
    padding: 30px 0 24px;
    border-bottom: 1px solid #0d2d1e;
    margin-bottom: 30px;
    position: relative;
}
.nexus-logo {
    font-family: 'Orbitron', monospace;
    font-size: 3.4rem;
    font-weight: 900;
    color: #00ffc8;
    letter-spacing: 6px;
    text-transform: uppercase;
    text-shadow:
        0 0 10px rgba(0,255,200,0.6),
        0 0 30px rgba(0,255,200,0.2),
        2px 2px 0 #ff00aa22;
    margin: 0;
    line-height: 1;
}
.nexus-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #ff00aa;
    letter-spacing: 5px;
    text-transform: uppercase;
    margin-top: 8px;
    text-shadow: 0 0 8px rgba(255,0,170,0.5);
}
.nexus-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: #1a4a3a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Glowing stat panels ── */
.neon-card {
    background: #050f0a;
    border: 1px solid #00ffc820;
    border-radius: 6px;
    padding: 20px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.neon-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ffc8, transparent);
}
.neon-card:hover { border-color: #00ffc840; }
.neon-val {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ffc8;
    text-shadow: 0 0 12px rgba(0,255,200,0.5);
    line-height: 1;
}
.neon-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #1e5a44;
    margin-top: 6px;
}
.neon-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.78rem;
    color: #2a6a50;
    margin-top: 2px;
}

/* ── Section divider ── */
.sec-divider {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 5px;
    text-transform: uppercase;
    color: #ff00aa;
    text-shadow: 0 0 6px rgba(255,0,170,0.4);
    border-bottom: 1px solid #1a0a14;
    padding-bottom: 6px;
    margin: 26px 0 16px;
}
.sec-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e0f0ff;
    letter-spacing: 2px;
    margin-bottom: 16px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #050a14;
    border: 1px solid #0a1e30;
    border-radius: 6px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #1a4a3a;
    font-family: 'Orbitron', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    border-radius: 4px;
    padding: 10px 16px;
}
.stTabs [aria-selected="true"] {
    background: #061a12 !important;
    color: #00ffc8 !important;
    text-shadow: 0 0 8px rgba(0,255,200,0.5);
    box-shadow: inset 0 0 12px rgba(0,255,200,0.05);
}

/* ── Prediction terminal ── */
.terminal-box {
    background: #020b06;
    border: 1px solid #00ffc830;
    border-radius: 6px;
    padding: 24px 28px;
    font-family: 'Share Tech Mono', monospace;
}
.terminal-result-churn {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    color: #ff2255;
    text-shadow: 0 0 16px rgba(255,34,85,0.6);
    letter-spacing: 3px;
}
.terminal-result-stay {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    color: #00ffc8;
    text-shadow: 0 0 16px rgba(0,255,200,0.6);
    letter-spacing: 3px;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent;
    border: 1px solid #00ffc8;
    color: #00ffc8;
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-radius: 4px;
    padding: 12px 20px;
    text-shadow: 0 0 8px rgba(0,255,200,0.4);
    box-shadow: 0 0 12px rgba(0,255,200,0.08);
    transition: all 0.2s;
}
.stButton > button:hover {
    background: rgba(0,255,200,0.08);
    box-shadow: 0 0 20px rgba(0,255,200,0.2);
}
.stDownloadButton > button {
    border-color: #ff00aa;
    color: #ff00aa;
    text-shadow: 0 0 8px rgba(255,0,170,0.4);
}

/* ── Sidebar ── */
.sidebar-head {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    letter-spacing: 4px;
    color: #00ffc8;
    text-transform: uppercase;
    text-shadow: 0 0 6px rgba(0,255,200,0.4);
    margin-bottom: 14px;
}

.stDataFrame { border: 1px solid #0a2a1e; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB CYBERPUNK THEME ─────────────────────────────────────────────────
CYBER_BG   = "#02080e"
NEON_GREEN = "#00ffc8"
NEON_PINK  = "#ff00aa"
NEON_BLUE  = "#00aaff"
NEON_YLW   = "#ffe600"
DARK_PANEL = "#040e0a"

plt.rcParams.update({
    "figure.facecolor":  CYBER_BG,
    "axes.facecolor":    DARK_PANEL,
    "axes.edgecolor":    "#0a2a1e",
    "axes.labelcolor":   "#2a6a50",
    "text.color":        "#8ab8a0",
    "xtick.color":       "#1a4a34",
    "ytick.color":       "#1a4a34",
    "grid.color":        "#061a10",
    "grid.linestyle":    "-",
    "grid.alpha":        1.0,
    "font.family":       "monospace",
})

NEON_CMAP = LinearSegmentedColormap.from_list(
    "neon", ["#020b06", "#003d28", "#00ffc8"], N=256
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nexus-header">
    <p class="nexus-logo">⚡ NEXUS</p>
    <p class="nexus-sub">// Churn Detection AI · Extra-Trees Ensemble · Neural Analytics</p>
    <p class="nexus-tag">B.Tech Gen AI · Semester II · Final Project · Customer Travel Dataset</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-head">⚡ System Config</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
    st.markdown("---")
    n_estimators  = st.slider("N Estimators",       50, 500, 200, 50)
    max_features  = st.select_slider("Max Features", options=["sqrt","log2","auto"], value="sqrt")
    min_samples   = st.slider("Min Samples Leaf",     1,  10,   2,  1)
    test_size     = st.slider("Test Split %",         10,  40,  20,  5) / 100
    random_state  = st.number_input("Seed", value=99, step=1)
    train_button  = st.button("▶ INITIALIZE TRAINING", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
                color:#1a4a34;line-height:2.2;letter-spacing:1px">
    SYS: ExtraTreesClassifier<br>
    TASK: Binary · Churn[0|1]<br>
    ENSEMBLE: Bagging Wrapper<br>
    CV: 5-Fold Stratified<br>
    MCC: Matthews Coeff<br><br>
    STATUS: ONLINE ◉
    </div>""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(f):
    return pd.read_csv(f)

if uploaded_file:
    df_raw = load_data(uploaded_file)
else:
    path = os.path.join(os.path.dirname(__file__), "Customertravel.csv")
    df_raw = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame({
        "Age": np.random.randint(25,55,300),
        "FrequentFlyer": np.random.choice(["Yes","No","No Record"],300),
        "AnnualIncomeClass": np.random.choice(["Low Income","Middle Income","High Income"],300),
        "ServicesOpted": np.random.randint(1,7,300),
        "AccountSyncedToSocialMedia": np.random.choice(["Yes","No"],300),
        "BookedHotelOrNot": np.random.choice(["Yes","No"],300),
        "Target": np.random.choice([0,1],300,p=[0.77,0.23]),
    })

df = df_raw.copy()

@st.cache_data
def preprocess(data):
    d = data.copy().dropna()
    encoders = {}
    for col in d.select_dtypes(include="object").columns:
        le = LabelEncoder()
        d[col] = le.fit_transform(d[col].astype(str))
        encoders[col] = le
    return d, encoders

df_clean, encoders = preprocess(df)
X = df_clean.drop(columns=["Target"])
y = df_clean["Target"]
feature_names = X.columns.tolist()

# ── TRAIN ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(n_est, mf, ms, t_size, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=t_size, random_state=seed, stratify=y)
    base = ExtraTreesClassifier(
        n_estimators=n_est, max_features=mf,
        min_samples_leaf=ms, random_state=seed,
        class_weight="balanced", bootstrap=True)
    clf = BaggingClassifier(estimator=base, n_estimators=5,
                            random_state=seed, max_samples=0.9)
    clf.fit(X_tr, y_tr)
    cv = cross_val_score(base, X, y, cv=5, scoring="f1")
    fi = np.mean([e.feature_importances_ for e in base.fit(X_tr,y_tr).estimators_], axis=0)
    return clf, X_tr, X_te, y_tr, y_te, cv, fi

if "model" not in st.session_state or train_button:
    with st.spinner("Initializing neural ensemble…"):
        model, X_train, X_test, y_train, y_test, cv_f1, feat_imp = train_model(
            n_estimators, max_features, min_samples, test_size, int(random_state))
    st.session_state.update({"model":model,"X_test":X_test,"y_test":y_test,
                              "cv_f1":cv_f1,"feat_imp":feat_imp})
    if train_button:
        st.success(f"⚡ System armed — CV F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
else:
    model    = st.session_state["model"]
    X_test   = st.session_state["X_test"]
    y_test   = st.session_state["y_test"]
    cv_f1    = st.session_state["cv_f1"]
    feat_imp = st.session_state["feat_imp"]

y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:,1]
acc         = accuracy_score(y_test, y_pred)
f1          = f1_score(y_test, y_pred)
mcc         = matthews_corrcoef(y_test, y_pred)
cm          = confusion_matrix(y_test, y_pred)
fpr,tpr,_   = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
report      = classification_report(y_test, y_pred, output_dict=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs(["SYS·INFO","DATA·SCAN","EVAL","SIGNALS","PREDICT"])

# ── TAB 1 – SYS INFO ──────────────────────────────────────────────────────────
with t1:
    st.markdown('<p class="sec-divider">// System Metrics</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,lbl,sub) in zip([c1,c2,c3,c4,c5],[
        (f"{len(df):,}",          "Records",    "data points loaded"),
        (str(len(feature_names)), "Features",   "input dimensions"),
        (f"{y.mean()*100:.1f}%",  "Churn Rate", "positive class ratio"),
        (f"{acc*100:.1f}%",       "Accuracy",   "test set performance"),
        (f"{mcc:.3f}",            "MCC Score",  "matthews correlation"),
    ]):
        col.markdown(f"""<div class="neon-card">
            <div class="neon-val">{val}</div>
            <div class="neon-label">{lbl}</div>
            <div class="neon-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-divider">// Raw Data Stream</p>', unsafe_allow_html=True)
    st.dataframe(df.head(14), use_container_width=True, height=330)

    col_s, col_q = st.columns(2)
    with col_s:
        st.markdown('<p class="sec-divider">// Statistical Profile</p>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    with col_q:
        st.markdown('<p class="sec-divider">// Data Quality Matrix</p>', unsafe_allow_html=True)
        q = pd.DataFrame({"Column":df.columns,"Type":df.dtypes.astype(str).values,
                           "Nulls":df.isnull().sum().values,"Unique":df.nunique().values})
        st.dataframe(q, use_container_width=True, hide_index=True)

# ── TAB 2 – DATA SCAN ─────────────────────────────────────────────────────────
with t2:
    st.markdown('<p class="sec-divider">// Feature Scan — Churn vs Retained</p>', unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 3, figsize=(16,9))
    fig.patch.set_facecolor(CYBER_BG)
    fig.suptitle("NEXUS DATA SCAN  ·  CHURN SIGNAL ANALYSIS",
                 fontsize=10, color=NEON_GREEN, fontfamily="monospace",
                 x=0.01, ha="left", y=1.02,
                 fontweight="bold")

    def neon_ax(ax):
        ax.set_facecolor(DARK_PANEL)
        ax.spines[["top","right","left","bottom"]].set_color("#0a2a1e")
        ax.tick_params(colors="#1a4a34", labelsize=8)
        ax.grid(True, color="#061a10", linewidth=0.8)

    # 1. Ring chart
    churn_c = df["Target"].value_counts()
    wedges,_,autos = axes[0,0].pie(
        churn_c, labels=["Retained","Churned"],
        autopct="%1.0f%%",
        colors=[NEON_BLUE, NEON_PINK],
        startangle=90,
        wedgeprops=dict(width=0.6, edgecolor=CYBER_BG, linewidth=3),
        textprops=dict(color="#2a6a50", fontsize=8, fontfamily="monospace"),
    )
    for a in autos:
        a.set_color("#e0f0ff"); a.set_fontsize(11); a.set_fontweight("bold")
    axes[0,0].set_title("CHURN SPLIT", fontsize=8, color=NEON_GREEN,
                         fontfamily="monospace", pad=10)

    # 2. Age — neon filled histogram
    for val,clr,lbl in [(0,NEON_BLUE,"Retained"),(1,NEON_PINK,"Churned")]:
        vals = df[df["Target"]==val]["Age"]
        n,bins,_ = axes[0,1].hist(vals, bins=14, alpha=0, density=True)
        x = (bins[:-1]+bins[1:])/2
        axes[0,1].fill_between(x, n, alpha=0.3, color=clr)
        axes[0,1].plot(x, n, color=clr, lw=2, label=lbl)
    axes[0,1].set_title("AGE DENSITY", fontsize=8, color=NEON_GREEN, fontfamily="monospace")
    axes[0,1].legend(fontsize=7)
    neon_ax(axes[0,1])

    # 3. Frequent flyer — neon bars
    ff_ct = df.groupby(["FrequentFlyer","Target"]).size().unstack(fill_value=0)
    x3 = np.arange(len(ff_ct)); w=0.35
    axes[0,2].bar(x3-w/2, ff_ct[0], w, color=NEON_BLUE,
                       edgecolor=CYBER_BG, alpha=0.85, label="Retained")
    axes[0,2].bar(x3+w/2, ff_ct[1], w, color=NEON_PINK,
                       edgecolor=CYBER_BG, alpha=0.85, label="Churned")
    axes[0,2].set_xticks(x3); axes[0,2].set_xticklabels(ff_ct.index, fontsize=7)
    axes[0,2].set_title("FREQUENT FLYER", fontsize=8, color=NEON_GREEN, fontfamily="monospace")
    axes[0,2].legend(fontsize=7)
    neon_ax(axes[0,2])

    # 4. Income — horizontal diverging
    inc_ct = df.groupby(["AnnualIncomeClass","Target"]).size().unstack(fill_value=0)
    y4 = np.arange(len(inc_ct))
    axes[1,0].barh(y4, inc_ct[0], color=NEON_BLUE, alpha=0.85, label="Retained", edgecolor=CYBER_BG)
    axes[1,0].barh(y4, -inc_ct[1], color=NEON_PINK, alpha=0.85, label="Churned", edgecolor=CYBER_BG)
    axes[1,0].set_yticks(y4); axes[1,0].set_yticklabels(inc_ct.index, fontsize=7)
    axes[1,0].axvline(0, color="#1a4a34", lw=1)
    axes[1,0].set_title("INCOME CLASS (DIVERGING)", fontsize=8, color=NEON_GREEN, fontfamily="monospace")
    axes[1,0].legend(fontsize=7)
    neon_ax(axes[1,0])

    # 5. Services — scatter density
    for val,clr in [(0,NEON_BLUE),(1,NEON_PINK)]:
        svc = df[df["Target"]==val]["ServicesOpted"]
        axes[1,1].scatter(svc + np.random.uniform(-0.2,0.2,len(svc)),
                          np.random.uniform(-0.3,0.3,len(svc)) + val,
                          color=clr, alpha=0.35, s=18, edgecolors="none")
    axes[1,1].set_yticks([0,1]); axes[1,1].set_yticklabels(["Retained","Churned"],fontsize=8)
    axes[1,1].set_xlabel("Services Opted", color="#2a6a50", fontsize=8)
    axes[1,1].set_title("SERVICES JITTER", fontsize=8, color=NEON_GREEN, fontfamily="monospace")
    neon_ax(axes[1,1])

    # 6. Hotel — stacked %
    hotel_ct = df.groupby(["BookedHotelOrNot","Target"]).size().unstack(fill_value=0)
    pct = hotel_ct.div(hotel_ct.sum(axis=1),axis=0)*100
    x6 = np.arange(len(pct))
    axes[1,2].bar(x6, pct[0], color=NEON_BLUE, alpha=0.85, label="Retained", edgecolor=CYBER_BG)
    axes[1,2].bar(x6, pct[1], bottom=pct[0], color=NEON_PINK,
                  alpha=0.85, label="Churned", edgecolor=CYBER_BG)
    axes[1,2].set_xticks(x6); axes[1,2].set_xticklabels(pct.index, fontsize=9)
    axes[1,2].set_ylabel("%", color="#2a6a50", fontsize=8)
    axes[1,2].set_title("HOTEL BOOKING %", fontsize=8, color=NEON_GREEN, fontfamily="monospace")
    axes[1,2].legend(fontsize=7)
    neon_ax(axes[1,2])

    plt.tight_layout()
    st.pyplot(fig)

    # Correlation
    st.markdown('<p class="sec-divider">// Correlation Matrix</p>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    fig2.patch.set_facecolor(CYBER_BG); ax2.set_facecolor(DARK_PANEL)
    corr = df_clean.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap=NEON_CMAP, linewidths=1.5, linecolor=CYBER_BG,
                ax=ax2, square=True,
                annot_kws={"size":9,"color":"#8ab8a0"},
                cbar_kws={"shrink":0.8})
    ax2.set_title("FEATURE CORRELATION MATRIX",
                  fontsize=10, color=NEON_GREEN, fontfamily="monospace", pad=14)
    ax2.tick_params(colors="#1a4a34", labelsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)

# ── TAB 3 – EVAL ──────────────────────────────────────────────────────────────
with t3:
    st.markdown('<p class="sec-divider">// Performance Readout</p>', unsafe_allow_html=True)
    e1,e2,e3,e4 = st.columns(4)
    for col,(val,lbl,sub) in zip([e1,e2,e3,e4],[
        (f"{acc*100:.2f}%",          "Accuracy",       "test set"),
        (f"{f1:.3f}",                "F1 Score",       "harmonic mean"),
        (f"{roc_auc:.3f}",           "ROC-AUC",        "area under curve"),
        (f"{mcc:.3f}",               "MCC",            "matthews coeff"),
    ]):
        col.markdown(f"""<div class="neon-card">
            <div class="neon-val" style="font-size:1.7rem">{val}</div>
            <div class="neon-label">{lbl}</div>
            <div class="neon-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown('<p class="sec-divider">// Confusion Matrix</p>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(5,4))
        fig3.patch.set_facecolor(CYBER_BG); ax3.set_facecolor(DARK_PANEL)
        sns.heatmap(cm, annot=True, fmt="d", cmap=NEON_CMAP,
                    xticklabels=["Retained","Churned"],
                    yticklabels=["Retained","Churned"],
                    linewidths=2, linecolor=CYBER_BG, ax=ax3,
                    annot_kws={"size":16,"weight":"bold","color":"#e0f8f0"})
        ax3.set_xlabel("PREDICTED", color="#2a6a50", fontsize=9)
        ax3.set_ylabel("ACTUAL",    color="#2a6a50", fontsize=9)
        ax3.set_title("CONFUSION MATRIX", color=NEON_GREEN,
                      fontsize=10, fontfamily="monospace")
        ax3.tick_params(colors="#1a4a34")
        fig3.tight_layout()
        st.pyplot(fig3)

    with col_roc:
        st.markdown('<p class="sec-divider">// ROC Curve</p>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(5,4))
        fig4.patch.set_facecolor(CYBER_BG); ax4.set_facecolor(DARK_PANEL)
        ax4.plot(fpr, tpr, color=NEON_GREEN, lw=2.5,
                 label=f"ExtraTrees  AUC={roc_auc:.3f}",
                 gapcolor=CYBER_BG)
        ax4.fill_between(fpr, tpr, alpha=0.06, color=NEON_GREEN)
        ax4.plot([0,1],[0,1], "--", color="#1a4a34", lw=1)
        ax4.set_xlabel("FPR", color="#2a6a50", fontsize=9)
        ax4.set_ylabel("TPR", color="#2a6a50", fontsize=9)
        ax4.set_title("ROC CURVE", color=NEON_GREEN, fontsize=10, fontfamily="monospace")
        ax4.spines[["top","right","left","bottom"]].set_color("#0a2a1e")
        ax4.legend(fontsize=9, facecolor=DARK_PANEL, labelcolor="#8ab8a0",
                   edgecolor="#0a2a1e")
        ax4.grid(True, color="#061a10")
        fig4.tight_layout()
        st.pyplot(fig4)

    # CV bars
    st.markdown('<p class="sec-divider">// 5-Fold Cross Validation · F1</p>', unsafe_allow_html=True)
    fig5, ax5 = plt.subplots(figsize=(9,2.5))
    fig5.patch.set_facecolor(CYBER_BG); ax5.set_facecolor(DARK_PANEL)
    folds = [f"F{i+1}" for i in range(len(cv_f1))]
    bar_colors = [NEON_GREEN if v==cv_f1.max() else NEON_BLUE for v in cv_f1]
    b5 = ax5.bar(folds, cv_f1, color=bar_colors, edgecolor=CYBER_BG, width=0.5)
    ax5.axhline(cv_f1.mean(), color=NEON_PINK, lw=1.5, ls="--",
                label=f"μ = {cv_f1.mean():.3f}")
    for bar,v in zip(b5, cv_f1):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{v:.3f}", ha="center", fontsize=9, color="#e0f8f0")
    ax5.set_title("CROSS VALIDATION · F1 SCORE", color=NEON_GREEN,
                  fontsize=9, fontfamily="monospace")
    ax5.spines[["top","right","left","bottom"]].set_color("#0a2a1e")
    ax5.tick_params(colors="#1a4a34"); ax5.legend(fontsize=9,facecolor=DARK_PANEL,
                                                   labelcolor="#8ab8a0",edgecolor="#0a2a1e")
    ax5.grid(True, axis="y", color="#061a10")
    fig5.tight_layout()
    st.pyplot(fig5)

    st.markdown('<p class="sec-divider">// Classification Report</p>', unsafe_allow_html=True)
    cr_df = pd.DataFrame(report).T.iloc[:-1].round(3)
    st.dataframe(
        cr_df.style.format("{:.3f}")
             .background_gradient(cmap="Greens", axis=0,
                                  subset=["precision","recall","f1-score"]),
        use_container_width=True)

    buf = io.BytesIO(); pickle.dump(model,buf); buf.seek(0)
    st.download_button("⬇ EXPORT MODEL.PKL", data=buf,
                       file_name="nexus_extratrees.pkl",
                       mime="application/octet-stream",
                       use_container_width=True)

# ── TAB 4 – SIGNALS ───────────────────────────────────────────────────────────
with t4:
    st.markdown('<p class="sec-divider">// Feature Signal Strength</p>', unsafe_allow_html=True)

    fi_df = pd.DataFrame({"Feature":feature_names,"Signal":feat_imp})\
              .sort_values("Signal", ascending=False).reset_index(drop=True)

    # Radar-style horizontal bar with glow effect
    fig6, ax6 = plt.subplots(figsize=(9,5))
    fig6.patch.set_facecolor(CYBER_BG); ax6.set_facecolor(DARK_PANEL)
    fi_s = fi_df.sort_values("Signal", ascending=True)
    clrs = [NEON_GREEN if i==len(fi_s)-1 else NEON_BLUE for i in range(len(fi_s))]
    bars6 = ax6.barh(fi_s["Feature"], fi_s["Signal"],
                     color=clrs, edgecolor=CYBER_BG, height=0.55, alpha=0.85)
    # Glow bars (wider, very transparent)
    ax6.barh(fi_s["Feature"], fi_s["Signal"],
             color=clrs, edgecolor="none", height=0.75, alpha=0.08)
    for bar,v in zip(bars6, fi_s["Signal"]):
        ax6.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                 f"{v:.4f}", va="center", fontsize=9, color="#8ab8a0")
    ax6.set_title("FEATURE SIGNAL STRENGTH · EXTRA TREES",
                  fontsize=10, color=NEON_GREEN, fontfamily="monospace", pad=14)
    ax6.set_xlabel("IMPORTANCE", color="#2a6a50", fontsize=9)
    ax6.spines[["top","right","left","bottom"]].set_color("#0a2a1e")
    ax6.tick_params(colors="#1a4a34"); ax6.grid(True, axis="x", color="#061a10")
    fig6.tight_layout()
    st.pyplot(fig6)

    fi_df["Rank"]  = range(1,len(fi_df)+1)
    fi_df["Share"] = (fi_df["Signal"]/fi_df["Signal"].sum()*100).round(1).astype(str)+"%"
    st.dataframe(
        fi_df[["Rank","Feature","Signal","Share"]]
          .style.format({"Signal":"{:.4f}"}),
        use_container_width=True, hide_index=True)

    top = fi_df.iloc[0]
    st.markdown(f"""
    <div style="background:#020b06;border:1px solid {NEON_GREEN}22;border-radius:6px;
                padding:16px 22px;margin-top:12px;font-family:'Share Tech Mono',monospace">
        <span style="color:{NEON_PINK};font-size:0.6rem;letter-spacing:4px">
        ▶ TOP SIGNAL DETECTED</span><br>
        <span style="color:{NEON_GREEN};font-size:1rem;text-shadow:0 0 8px rgba(0,255,200,0.4)">
        {top['Feature']}</span>
        <span style="color:#2a6a50;font-size:0.82rem">
         carries {top['Share']} of predictive weight</span>
    </div>
    """, unsafe_allow_html=True)

# ── TAB 5 – PREDICT ───────────────────────────────────────────────────────────
with t5:
    st.markdown('<p class="sec-divider">// Live Passenger Inference</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Input Passenger Profile</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age    = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 34)
        ff     = st.selectbox("Frequent Flyer", ["No","Yes","No Record"])
        income = st.selectbox("Annual Income Class",
                               ["Low Income","Middle Income","High Income"])
    with col2:
        services = st.slider("Services Opted", 1, 6, 3)
        social   = st.selectbox("Social Media Sync", ["No","Yes"])
        hotel    = st.selectbox("Hotel Booked", ["No","Yes"])

    if st.button("▶ EXECUTE PREDICTION", use_container_width=True):
        inp = {"Age":age,"FrequentFlyer":ff,"AnnualIncomeClass":income,
               "ServicesOpted":services,"AccountSyncedToSocialMedia":social,
               "BookedHotelOrNot":hotel}
        inp_df = pd.DataFrame([inp])
        for c in inp_df.select_dtypes(include="object").columns:
            if c in encoders:
                le = encoders[c]
                v = inp_df[c].astype(str).values[0]
                inp_df[c] = le.transform([v])[0] if v in le.classes_ else 0

        pred  = model.predict(inp_df[feature_names])[0]
        proba = model.predict_proba(inp_df[feature_names])[0]

        r1, r2 = st.columns(2)
        if pred == 1:
            r1.markdown(f"""
            <div class="terminal-box">
                <div style="color:{NEON_PINK};font-size:0.6rem;letter-spacing:4px;
                            font-family:'Share Tech Mono',monospace;margin-bottom:10px">
                ⚠ ALERT · CHURN RISK DETECTED</div>
                <div class="terminal-result-churn">WILL CHURN</div>
            </div>""", unsafe_allow_html=True)
        else:
            r1.markdown(f"""
            <div class="terminal-box">
                <div style="color:{NEON_GREEN};font-size:0.6rem;letter-spacing:4px;
                            font-family:'Share Tech Mono',monospace;margin-bottom:10px">
                ✓ CLEAR · RETENTION PREDICTED</div>
                <div class="terminal-result-stay">WILL STAY</div>
            </div>""", unsafe_allow_html=True)

        r2.markdown(f"""
        <div class="neon-card" style="text-align:center;padding:30px">
            <div class="neon-val" style="font-size:2.6rem;
                color:{'#ff2255' if pred==1 else '#00ffc8'}">
                {proba[1]*100:.1f}%
            </div>
            <div class="neon-label">Churn Probability</div>
            <div class="neon-sub">Retention: {proba[0]*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        # Neon probability bars
        fig_p, ax_p = plt.subplots(figsize=(7,1.4))
        fig_p.patch.set_facecolor(CYBER_BG); ax_p.set_facecolor(CYBER_BG)
        ax_p.barh(["Retained"], [proba[0]*100], color=NEON_BLUE,  alpha=0.85,
                  edgecolor=CYBER_BG, height=0.45)
        ax_p.barh(["Churned"],  [proba[1]*100], color=NEON_PINK,  alpha=0.85,
                  edgecolor=CYBER_BG, height=0.45)
        for val,cat,clr in [(proba[0]*100,"Retained",NEON_BLUE),
                             (proba[1]*100,"Churned", NEON_PINK)]:
            ax_p.text(min(val+1,95), ["Retained","Churned"].index(cat),
                      f"{val:.1f}%", va="center", fontsize=10, color="#e0f8f0")
        ax_p.set_xlim(0,108); ax_p.spines[["top","right","left","bottom"]].set_visible(False)
        ax_p.tick_params(colors="#1a4a34",labelsize=9); ax_p.set_xticks([])
        ax_p.set_title("PROBABILITY READOUT", fontsize=8, color=NEON_GREEN,
                        fontfamily="monospace", loc="left")
        fig_p.tight_layout()
        st.pyplot(fig_p)

    # Batch
    st.markdown('<p class="sec-divider">// Batch Inference Module</p>', unsafe_allow_html=True)
    batch_file = st.file_uploader("Upload CSV", type="csv", key="batch")
    if batch_file:
        bdf = pd.read_csv(batch_file); bp = bdf.copy()
        for c in bp.select_dtypes(include="object").columns:
            if c in encoders:
                le = encoders[c]
                bp[c] = bp[c].astype(str).apply(
                    lambda v: le.transform([v])[0] if v in le.classes_ else 0)
        avail = [f for f in feature_names if f in bp.columns]
        if avail:
            preds_b = model.predict(bp[avail])
            proba_b = model.predict_proba(bp[avail])[:,1]
            bdf["Churn_Prediction"] = preds_b
            bdf["Churn_Prob_%"]     = (proba_b*100).round(2)
            bdf["Status"]           = bdf["Churn_Prediction"].map({0:"RETAINED",1:"AT RISK"})
            at_risk = preds_b.sum()
            st.success(f"⚡ Processed {len(bdf)} records — {at_risk} flagged AT RISK ({at_risk/len(bdf)*100:.1f}%)")
            st.dataframe(bdf, use_container_width=True)
            st.download_button("⬇ DOWNLOAD RESULTS",
                               bdf.to_csv(index=False).encode("utf-8"),
                               "nexus_batch_output.csv","text/csv",
                               use_container_width=True)
        else:
            st.error("COLUMN MISMATCH — ensure CSV matches training feature schema.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #0a2a1e;margin-top:40px;padding-top:14px;
            font-family:'Share Tech Mono',monospace;font-size:0.58rem;
            color:#1a4a34;letter-spacing:3px;text-align:center">
    NEXUS CHURN AI · EXTRA-TREES ENSEMBLE · B.TECH GEN AI · SEM II ·
    STATUS: ONLINE ◉ · ALL SYSTEMS NOMINAL
</div>
""", unsafe_allow_html=True)
