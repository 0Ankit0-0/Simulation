<div align="center">

# ⚖️ JURIX
### AI-Powered Courtroom Simulation Platform

*Simulate. Argue. Learn. Decide.*

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Built With](https://img.shields.io/badge/built%20with-React%20%2B%20Flask-informational)
![Status](https://img.shields.io/badge/status-active-brightgreen)

</div>

---

## 📌 Overview

**Jurix** is an interactive, AI-driven courtroom simulation platform that replicates real-world legal proceedings. Users can upload case files, evidence, and documents — then watch as AI agents representing the **Prosecutor**, **Defense**, and **Judge** engage in structured legal arguments.

> Ideal for law students, educators, and legal professionals looking to practice argumentation and study courtroom dynamics in a safe, controlled environment.

---

## ✨ Features

### 🤖 AI Courtroom Agents
| Agent | Role |
|-------|------|
| **ProsecutorAgent** | Builds prosecutorial arguments using structured legal datasets |
| **DefenseAgent** | Formulates defense strategies and counter-arguments |
| **JudgeAgent** | Evaluates both sides, maintains courtroom flow, and renders verdicts |

---

### 📂 Evidence Parsing
- Upload **PDFs, DOCX, images, or plain text** documents
- AI parser converts raw evidence into structured, readable summaries
- Review, approve, or manually edit parsed evidence before simulation begins

---

### 📚 Legal Knowledge Base
- Covers **IPC**, **CrPC**, **Constitution of India**, and other Indian legal frameworks
- Agents reason through arguments using domain-specific structured datasets

---

### 🔄 Simulation Workflow

```
Upload Case Files & Evidence
         ↓
AI Parses & Summarizes Evidence
         ↓
Review & Approve Evidence
         ↓
Launch Courtroom Simulation
         ↓
Interactive AI Agent Proceedings
         ↓
Detailed Verdict + Case Summary Export
```

---

### 🧠 Multi-Tier Fallback AI Architecture

```
Primary   →  Local custom-trained legal model
Secondary →  Pre-trained open-source models
Tertiary  →  OpenAI / Gemini API (enhanced reasoning)
```

Ensures uninterrupted simulation even when primary models are unavailable.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + Tailwind CSS |
| Backend | Flask (Python) |
| Storage | JSON / MongoDB |
| AI Layer | Custom Legal Model + OpenAI / Gemini Fallback |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/jurix.git
cd jurix
```

### 2. Setup Python Environment
```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Run the Backend
```bash
cd backend
python app.py
```

### 4. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

### 5. Open in Browser
```
http://localhost:5173
```

---

## 🖥️ Usage

1. **Upload** case evidence via the `CaseSubmit` page
2. **Review & approve** parsed evidence in `CaseReview`
3. **Launch** the simulation to interact with AI agents
4. **Observe** AI-generated legal arguments and verdicts
5. **Export** summaries for study, training, or future reference

---

## 🖼️ Screenshots

| Dashboard | Case Cards | New Case Submission | Legal Assistant Chat |
|:---------:|:----------:|:-------------------:|:--------------------:|
| ![Dashboard](Image%201.png) | ![Case Cards](Image%202.png) | ![New Case](Image%203.png) | ![Chat](Image%204.png) |

---

## 📊 Datasets Used

- Indian Penal Code (IPC)
- Code of Criminal Procedure (CrPC)
- Constitution of India
- Mock cases and legal dialogues (custom training data)

---

## 🤝 Contributing

Contributions are welcome and appreciated!

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 🏆 Recognition

Built as a **solo endeavor**, Jurix was later presented at **Avishkar 2025** — a competitive inter-collegiate research festival — as a working simulation of AI-assisted judicial reasoning.

> **Special thanks to:**
> 
> 🙏 **Dhruv Suthar** — Co-presenter & collaborator  
> 🙏 **Zoya Khan** — Co-presenter & collaborator

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<sub>Built with ⚖️ and a lot of legal caffeine by Ankit</sub>
</div>
```

---

- Improved the Git contribution steps with actual commands
- Rewrote the Avishkar section with better formatting and emoji credits
- Added a footer byline for personal branding
