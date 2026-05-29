import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import os

st.set_page_config(
    page_title="Astronomical Image Classifier",
    layout="wide",
)

MAIN_CLASSES = [
    'Merging Galaxy',
    'Elliptical Galaxy',
    'Spiral Galaxy',
    'Edge-on Galaxy',
    'Nebula',
    'Planetary Object',
    'Star Cluster',
]

PLANET_CLASSES = [
    'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter',
    'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Moon'
]

NEBULA_CLASSES = [
    'Dark Nebula',
    'Emission Nebula',
    'Planetary Nebula',
    'Reflection Nebula',
    'Supernova Remnants',
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource
def load_models():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    main_model = build_model(7)
    main_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'phase2_v2_best.pth')
    main_model.load_state_dict(torch.load(main_model_path, map_location=device))
    main_model.to(device).eval()

    planet_model = build_model(10)
    planet_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'planet_phase2_best.pth')
    planet_model.load_state_dict(torch.load(planet_model_path, map_location=device))
    planet_model.to(device).eval()

    nebula_model = build_model(5)
    nebula_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'nebula_phase2_best.pth')
    nebula_model.load_state_dict(torch.load(nebula_model_path, map_location=device))
    nebula_model.to(device).eval()

    return main_model, planet_model, nebula_model, device

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.features[-1][0]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

def make_gradcam_overlay(img_pil, cam):
    img_resized = img_pil.resize((224, 224))
    img_np = np.array(img_resized) / 255.0
    heatmap = cm.jet(cam)[:, :, :3]
    overlay = 0.55 * img_np + 0.45 * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

def predict(img_pil, main_model, planet_model, nebula_model, device):
    transform = get_transform()
    img_np = np.array(img_pil.convert('RGB'), dtype=np.uint8)
    tensor = transform(Image.fromarray(img_np)).unsqueeze(0).to(device)

    # Main prediction
    with torch.no_grad():
        outputs = main_model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        top3 = probs.topk(3)

    # Grad-CAM
    gradcam = GradCAM(main_model)
    cam = gradcam.generate(tensor, pred_idx)
    overlay = make_gradcam_overlay(img_pil, cam)

    # Sub-classification
    sub_pred = None
    sub_confidence = None
    sub_top3 = None

    if MAIN_CLASSES[pred_idx] == 'Planetary Object':
        with torch.no_grad():
            sub_out = planet_model(tensor)
            sub_probs = torch.softmax(sub_out, dim=1)[0]
            sub_idx = sub_probs.argmax().item()
            sub_confidence = sub_probs[sub_idx].item()
            sub_pred = PLANET_CLASSES[sub_idx]
            sub_top3 = sub_probs.topk(3)
            sub_classes = PLANET_CLASSES

    elif MAIN_CLASSES[pred_idx] == 'Nebula':
        with torch.no_grad():
            sub_out = nebula_model(tensor)
            sub_probs = torch.softmax(sub_out, dim=1)[0]
            sub_idx = sub_probs.argmax().item()
            sub_confidence = sub_probs[sub_idx].item()
            sub_pred = NEBULA_CLASSES[sub_idx]
            sub_top3 = sub_probs.topk(3)
            sub_classes = NEBULA_CLASSES

    return {
        'pred_class': MAIN_CLASSES[pred_idx],
        'confidence': confidence,
        'top3_probs': top3.values.tolist(),
        'top3_indices': top3.indices.tolist(),
        'overlay': overlay,
        'sub_pred': sub_pred,
        'sub_confidence': sub_confidence,
        'sub_top3': sub_top3,
        'sub_classes': sub_classes if sub_pred else None,
    }

st.title("Astronomical Image Classifier")
st.markdown("Upload an image of a galaxy, nebula, planet, or star cluster to classify it.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This classifier uses **EfficientNet-B0** trained on:
    - Galaxy10 DECaLS dataset
    - Kaggle space images
    
    **7 main classes:**
    - Merging Galaxy
    - Elliptical Galaxy
    - Spiral Galaxy
    - Edge-on Galaxy
    - Nebula
    - Planetary Object
    - Star Cluster
    
    **Sub-classifiers:**
    - Planetary → 10 planet/moon types
    -  Nebula → 5 nebula types
    
    **Accuracy:** 93.25% (main model)
    """)
    st.divider()
    st.markdown("Built with PyTorch + Streamlit")

# Main area
uploaded_file = st.file_uploader(
    "Choose an astronomical image",
    type=['jpg', 'jpeg', 'png'],
)

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    with st.spinner("Classifying..."):
        main_model, planet_model, nebula_model, device = load_models()
        results = predict(img, main_model, planet_model, nebula_model, device)

    with col2:
        st.subheader("Grad-CAM — What the model sees")
        st.image(results['overlay'], use_container_width=True)

    st.divider()

    # Main prediction
    st.subheader("Classification Results")

    conf_pct = results['confidence'] * 100
    if conf_pct >= 90:
        conf_color = "🟢"
    elif conf_pct >= 70:
        conf_color = "🟡"
    else:
        conf_color = "🔴"

    st.markdown(f"## {conf_color} {results['pred_class']}")
    st.progress(results['confidence'])
    st.markdown(f"**Confidence: {conf_pct:.1f}%**")

    # Sub-classification
    if results['sub_pred']:
        st.divider()
        icon = "" if results['pred_class'] == 'Planetary Object' else ""
        st.markdown(f"### {icon} Sub-classification: **{results['sub_pred']}**")
        st.progress(results['sub_confidence'])
        st.markdown(f"Confidence: {results['sub_confidence']*100:.1f}%")

        # Sub top 3
        st.markdown("**Top 3 sub-type predictions:**")
        for prob, idx in zip(results['sub_top3'].values, results['sub_top3'].indices):
            st.markdown(f"- {results['sub_classes'][idx]}: {prob*100:.1f}%")

    st.divider()

    # Top 3 main predictions
    st.markdown("**Top 3 main predictions:**")
    for prob, idx in zip(results['top3_probs'], results['top3_indices']):
        bar_val = prob
        st.markdown(f"**{MAIN_CLASSES[idx]}** — {prob*100:.1f}%")
        st.progress(bar_val)