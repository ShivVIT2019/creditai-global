---
title: CreditAI Global
emoji: 💳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# 💳 CreditAI Global — Smart Credit Approval System

AI-powered dual-market credit assessment system for **USA 🇺🇸** and **India 🇮🇳**, with a built-in AI pricing agent.

🔗 **Live Demo:** [huggingface.co/spaces/sivasaiatchyut/creditai-global](https://huggingface.co/spaces/sivasaiatchyut/creditai-global)

---

## 🎥 Walkthrough

### USA Credit Application
![USA Credit Application](./docs/images/1.png)

### USA — Full Form with Prediction
![USA Full View](./docs/images/3.png)

### USA — Assessment Result (Approved, Medium Risk)
![USA Assessment Detail](./docs/images/4.png)

### India Credit Application (HDFC Credila Style)
![India Credit Application](./docs/images/5.png)

### India — Assessment Result (Approved, ₹52,954 Credit Limit)
![India Assessment Detail](./docs/images/2.png)

---

## 🎯 What It Does

| Market | Scoring System | Verification | Supported Cards |
|---|---|---|---|
| 🇺🇸 USA | FICO + Random Forest (86.5% AUC-ROC) | SSN | Visa, Mastercard, Amex, Discover |
| 🇮🇳 India | CIBIL + regional risk | PAN & Aadhaar | Visa, Mastercard, RuPay, Amex |

**AI Pricing Agent:** multi-factor weighted optimization (risk 60% / market 30% / profit 10%) → applicant-specific credit limits and interest rates with explainable reasoning.

---

## 🏗 Architecture

```mermaid
flowchart TD
    UI[Gradio UI<br/>Tabs: Credit Risk - AI Pricing Agent] --> ROUTE{Select Country}
    ROUTE -->|USA| USA[USA Pipeline<br/>Random Forest - 2.2M loans]
    ROUTE -->|India| IND[India Pipeline<br/>CIBIL + Regional Tiers]
    USA --> AGENT[AI Pricing Agent<br/>Risk 60% - Market 30% - Profit 10%]
    IND --> AGENT
    AGENT --> OUT[Decision + Credit Limit + APR + Explanation]
```

---

## 🧠 Technology Stack

- **ML:** scikit-learn (Random Forest Classifier), joblib
- **UI:** Gradio 4.31.0
- **Data:** 2.2M Lending Club loans (training), pandas, numpy
- **Deployment:** Hugging Face Spaces (free tier, Python 3.11)
- **Model artifacts:** stored via Git LFS

---

## 📊 Model Performance

- **ROC-AUC:** 86.5%
- **Training set:** 2.2M loan records from Lending Club
- **Features:** 17+ engineered features (DTI, utilization, delinquency count, account age, inquiries, bankruptcy indicators)
- **Preprocessing pipeline:** imputer + scaler, persisted via joblib

---

## 🚀 Run Locally

```bash
git clone https://github.com/ShivVIT2019/creditai-global.git
cd creditai-global
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 in your browser.

---

## 📦 Project Structure
creditai-global/
├── app.py # Gradio entry point
├── agent_tab.py # AI pricing agent UI
├── agent/ # Pricing agent logic
├── ml/
│ └── artifacts/ # Trained model + preprocessors (Git LFS)
├── docs/images/ # Screenshots for README
├── requirements.txt
└── README.md

---

## 🌐 Deployment Notes

Deployed on **Hugging Face Spaces** (free tier) after migrating from Google Cloud Run. Full deployment walkthrough: [`DEPLOYMENT.md`](./DEPLOYMENT.md).

---

## 👨‍💻 Author

**Sivasai Atchyut Akella** (Atchu)
MS Computer Science (AI concentration), Binghamton University
- 🔗 Portfolio: [portfolio-nine-rho-36.vercel.app](https://portfolio-nine-rho-36.vercel.app)
- 💼 LinkedIn: [linkedin.com/in/atchyut](https://linkedin.com/in/atchyut)
- 💻 GitHub: [@ShivVIT2019](https://github.com/ShivVIT2019)

---

## 📄 License

MIT License

> ⚠️ Research and portfolio demonstration. Not a substitute for professional credit decisioning or regulated financial advice.
