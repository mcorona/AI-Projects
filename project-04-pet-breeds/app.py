"""
Streamlit dashboard for the pet-breed classifier.

Three tabs:
  Classify   upload an image, get top-5 breeds with confidence, and see
             whether the predicted breed is one ImageNet already knew.
  Results    the baseline ladder and the held-out test numbers.
  ImageNet   the overlap analysis this project exists to make.

Reads a trained checkpoint and precomputed reports; never trains.

Author: Manuel Corona
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "output" / "reports"
MODELS = ROOT / "output" / "models"

st.set_page_config(page_title="Pet Breeds", page_icon="::", layout="wide")

MODEL_LABELS = {
    "majority": "Majority class (floor)",
    "classical": "HOG + colour histogram + linear SVM",
    "zero_shot": "ImageNet classifier, zero-shot",
    "linear_probe": "Linear probe on frozen features",
    "finetuned": "Fine-tuned ResNet-50",
}
LADDER = ["majority", "classical", "zero_shot", "linear_probe", "finetuned"]


@st.cache_data
def load_report(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def all_metrics():
    out = {}
    for name in ("baseline_metrics.json", "model_metrics.json"):
        r = load_report(name)
        if r:
            out.update(r)
    return out


@st.cache_resource(show_spinner="Loading the fine-tuned model ...")
def load_model():
    import torch
    from src.data.loader import pick_device
    from src.models.train import load_finetuned
    device = pick_device() if torch.backends.mps.is_available() else "cpu"
    model, ckpt = load_finetuned(device)
    return model, ckpt, device


@st.cache_data
def class_info():
    from src.data.imagenet_overlap import in_imagenet
    from src.data.loader import class_names, species_of
    names = class_names()
    return names, in_imagenet(names), species_of(names).astype(bool)


# --- Tab 1: classify ------------------------------------------------------

def tab_classify():
    names, known, is_cat = class_info()

    if not (MODELS / "resnet50_finetuned.pt").exists():
        st.warning("No checkpoint found. Run "
                   "`python -m src.models.run_training --stage finetune` first.")
        return

    uploaded = st.file_uploader("Upload a photo of a cat or dog",
                                type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to classify it into one of the 37 breeds.")
        st.caption(
            "The model only knows these 37 breeds. Given anything else — a "
            "different breed, a rabbit, a sofa — it will still return its most "
            "confident guess among the 37. That is a real limitation, not a bug, "
            "and it is why the confidence numbers matter as much as the ranking."
        )
        return

    from PIL import Image
    import torch
    from src.data.loader import eval_transform

    img = Image.open(uploaded).convert("RGB")
    model, ckpt, device = load_model()
    x = eval_transform()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].float().cpu().numpy()

    left, right = st.columns([1, 1.4])
    with left:
        st.image(img, use_container_width=True)
    with right:
        order = np.argsort(-probs)[:5]
        st.subheader(f"{names[order[0]]}")
        st.caption(f"confidence {probs[order[0]]:.1%}"
                   + ("  ·  cat" if is_cat[order[0]] else "  ·  dog"))
        rows = [{"Breed": names[i], "Confidence": f"{probs[i]:.1%}",
                 "In ImageNet-1k": "yes" if known[i] else "no"} for i in order]
        st.table(pd.DataFrame(rows).set_index("Breed"))
        if probs[order[0]] < 0.5:
            st.warning(
                "Low confidence. On the held-out test set this model is much less "
                "reliable below 50% confidence — see the calibration figures in "
                "the Results tab."
            )


# --- Tab 2: results -------------------------------------------------------

def tab_results():
    metrics = all_metrics()
    if not metrics:
        st.warning("No reports yet. Run the baselines and the evaluation first.")
        return

    st.markdown("**Oxford-IIIT Pet, official test split** — 3,669 images, 37 breeds. "
                "Every number below comes from one pass over that split.")
    rows = []
    for key in LADDER:
        m = metrics.get(key)
        if not m:
            continue
        rows.append({
            "Model": MODEL_LABELS[key],
            "Accuracy": m["accuracy"],
            "Macro F1": m["macro_f1"],
            "Top-5": m.get("top5_accuracy"),
            "In ImageNet": m["breeds_in_imagenet"]["accuracy"],
            "Not in ImageNet": m["breeds_not_in_imagenet"]["accuracy"],
        })
    df = pd.DataFrame(rows)
    for c in ("Accuracy", "Macro F1", "Top-5", "In ImageNet", "Not in ImageNet"):
        df[c] = df[c].map(lambda v: "—" if v is None or pd.isna(v) else f"{v:.4f}")
    st.table(df.set_index("Model"))

    ft = metrics.get("finetuned")
    if ft and "calibration" in ft:
        st.subheader("Is the confidence trustworthy?")
        c1, c2, c3 = st.columns(3)
        cal = ft["calibration"]
        c1.metric("Expected calibration error", f"{cal['ece']:.4f}")
        c2.metric("Mean confidence", f"{cal['mean_confidence']:.1%}")
        c3.metric("Accuracy", f"{cal['accuracy']:.1%}")
        st.caption(
            "A model that says 90% should be right 90% of the time. ECE is the "
            "average gap between the two, weighted by how many predictions land "
            "in each confidence band."
        )
        st.dataframe(pd.DataFrame(ft["calibration_bins"]), hide_index=True)

    if ft:
        st.subheader("Where it still fails")
        st.caption("The most frequent (true → predicted) mistakes on the test split.")
        st.table(pd.DataFrame(ft["top_confusions"]).set_index("true"))


# --- Tab 3: ImageNet overlap ---------------------------------------------

def tab_imagenet():
    names, known, is_cat = class_info()
    metrics = all_metrics()

    st.markdown(
        "ImageNet-1k spends about 120 of its 1,000 classes on dog breeds and "
        "5 on domestic cats. That lopsidedness turns this dataset into a natural "
        "experiment: **every model here that starts from ImageNet weights has "
        "already been trained on a label set containing most of the dogs and "
        "almost none of the cats.**"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Breeds ImageNet already names", f"{known.sum()}/{len(names)}")
    c2.metric("Dog breeds", f"{(known & ~is_cat).sum()}/{(~is_cat).sum()}")
    c3.metric("Cat breeds", f"{(known & is_cat).sum()}/{is_cat.sum()}")

    rows = []
    for key in LADDER:
        m = metrics.get(key)
        if not m:
            continue
        a, b = m["breeds_in_imagenet"]["accuracy"], m["breeds_not_in_imagenet"]["accuracy"]
        rows.append({"Model": MODEL_LABELS[key],
                     "In ImageNet": f"{a:.4f}", "Not in ImageNet": f"{b:.4f}",
                     "Gap": f"{a - b:+.4f}"})
    if rows:
        st.subheader("Accuracy split by whether ImageNet already knew the breed")
        st.table(pd.DataFrame(rows).set_index("Model"))

    st.subheader("The 37 breeds")
    st.dataframe(pd.DataFrame({
        "Breed": names,
        "Species": ["cat" if c else "dog" for c in is_cat],
        "In ImageNet-1k": ["yes" if k else "no" for k in known],
    }), hide_index=True, height=420)


st.title("Pet breeds, and what ImageNet already knew")
st.caption("Fine-grained classification of 37 cat and dog breeds — with the "
           "pretraining overlap measured instead of ignored.")

tabs = st.tabs(["Classify", "Results", "ImageNet overlap"])
with tabs[0]:
    tab_classify()
with tabs[1]:
    tab_results()
with tabs[2]:
    tab_imagenet()
