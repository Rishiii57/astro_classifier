# Demo

## Hosted Demo

Live web application deployed on Streamlit Community Cloud:

[https://astroclassifier-fvnzkywhwr6bqmpcyfvisz.streamlit.app/](https://astroclassifier-fvnzkywhwr6bqmpcyfvisz.streamlit.app/)

Upload any astronomical image — galaxy, nebula, planet, or star cluster — to get:
- Main classification with confidence score
- Sub-classification for planetary objects (10 types) and nebulae (5 types)
- Grad-CAM heatmap showing what the model focused on
- AI-generated description with a NASA/Wikipedia learn more link

---

## Demo Video

Full walkthrough of the application demonstrating classification of a planet, nebula, and galaxy:

[YouTube : Astronomical Image Classifier Live Demo](https://www.youtube.com/watch?v=iJ-GqAHppoM)

---

## Local Inference

To run the app locally:

```bash
git clone https://github.com/Rishiii57/astro_classifier.git
cd astro_classifier
pip install -r requirements.txt
streamlit run app/app.py
```

Add a `.env` file in the project root with your Groq API key for AI descriptions:

```
GROQ_API_KEY=your_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).
