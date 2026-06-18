# Long Hair Identification — ML Gender Classifier

A machine learning web application that classifies gender based on **hair length** for individuals aged 20–30, and by **actual gender** for individuals outside that range. The image analysis now runs through a small Python backend that keeps the model key in the server environment and can talk to Anthropic or an OpenAI-compatible endpoint.

---

## 🧠 Decision Logic

| Age Range       | Hair Length | Actual Gender | Predicted Gender | Rule Applied   |
|-----------------|-------------|---------------|------------------|----------------|
| 20–30           | Long        | Male          | **Female**       | Hair-based     |
| 20–30           | Long        | Female        | **Female**       | Hair-based     |
| 20–30           | Short       | Female        | **Male**         | Hair-based     |
| 20–30           | Short       | Male          | **Male**         | Hair-based     |
| < 20 or > 30    | Any         | Male          | **Male**         | Actual gender  |
| < 20 or > 30    | Any         | Female        | **Female**       | Actual gender  |

---

## 🚀 Getting Started

### Prerequisites

- A modern web browser (Chrome, Firefox, Edge, Safari)
- Either an **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com) -or- an **OpenAI-compatible key** for a vision-capable Nemotron endpoint

### Running the App

1. **Clone or unzip** the project folder.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set `MODEL_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`, or `MODEL_PROVIDER=openai` with `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_API_URL` for a Nemotron/OpenAI-compatible endpoint.
4. Start the backend with `python app.py`.
5. Open `http://127.0.0.1:5000` in your browser.
6. Upload a photo (JPG/PNG/WEBP) of a person with a clearly visible face and hair.
7. Click **Analyse Image** and view the results.

---

## 📁 Project Structure

```
long-hair-identification/
├── app.py          # Flask backend that calls Anthropic or an OpenAI-compatible API
├── index.html      # Main GUI layout
├── requirements.txt # Python dependencies
├── style.css       # Full styling (sidebar, cards, result UI)
├── app.js          # ML logic, API integration, result rendering
└── README.md       # This file
```

---

## ⚙️ How It Works

```
Upload Photo
     │
     ▼
Vision model analyses:
  • Estimated age + age range
  • Hair length (long / short)
  • Actual biometric gender
     │
     ▼
Decision Engine:
  ┌─────────────────────────────────────────────────────┐
  │  Age 20–30?                                         │
  │  YES → predict by hair length                       │
  │         long hair → Female                          │
  │         short hair → Male                           │
  │  NO  → predict actual gender (hair irrelevant)      │
  └─────────────────────────────────────────────────────┘
     │
     ▼
Result Card:
  • Predicted gender
  • Rule applied + explanation
  • Confidence scores (overall, actual gender, hair length)
  • Plain-English model reasoning
```

---

## 🔑 API Configuration

This project defaults to the **Anthropic Claude claude-sonnet-4-6** model with vision capabilities, but it can also be pointed at an OpenAI-compatible endpoint if your Nemotron key is exposed that way.

- Anthropic model: `claude-sonnet-4-6`
- Anthropic endpoint: `https://api.anthropic.com/v1/messages`
- OpenAI-compatible endpoint: set `MODEL_PROVIDER=openai` and configure `OPENAI_API_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY`. For NVIDIA's hosted Nemotron endpoint, use `OPENAI_API_URL=https://integrate.api.nvidia.com/v1` and include the provider namespace in the model id, for example `OPENAI_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`.
- Your API key is stored in the local backend `.env` file and is never entered in the browser.

---

## 📊 Output Fields

| Field                   | Description                                      |
|-------------------------|--------------------------------------------------|
| `estimatedAge`          | Detected age of the person                       |
| `ageRange`              | Estimated age range (e.g. "22–27")              |
| `hairLength`            | `long` or `short`                                |
| `hairLengthDetail`      | Description of hair style                        |
| `actualGender`          | Biometric gender detected                        |
| `actualGenderConfidence`| Confidence of actual gender (0–100)              |
| `predictedGender`       | Final prediction per decision logic              |
| `ruleApplied`           | `hair-based` or `actual-gender`                  |
| `overallConfidence`     | Overall model confidence (0–100)                 |
| `reasoning`             | Plain English explanation                        |

---

## 🎯 Accuracy Notes

- Results depend on image quality — ensure the face and hair are clearly visible.
- The model estimates age; borderline cases (e.g. age 19 or 31) may vary slightly.
- For best results, use well-lit, front-facing portrait-style photos.

---

## 👨‍💻 Technologies Used

- **HTML5 / CSS3 / Vanilla JavaScript** — no frameworks or build tools
- **Python / Flask** — backend API for image analysis
- **Anthropic Claude or an OpenAI-compatible vision API** — for image analysis
- **Font Awesome 6** — icon set
- Fully responsive layout (sidebar collapses on mobile)

---

## 📄 License

MIT License — free to use, modify, and distribute.
