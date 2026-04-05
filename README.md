# ⚖️ Jurix — AI Courtroom Simulation Platform

**Jurix** is an AI-driven courtroom simulation system that replicates real-world judicial proceedings using autonomous legal agents. It enables users to analyze cases, test arguments, and understand courtroom dynamics through structured, role-based AI reasoning.

Designed for **law students, researchers, and legal-tech innovators**, Jurix provides a controlled environment to simulate litigation workflows—from evidence submission to final verdict.

---

## 🚀 Core Capabilities

### 🧠 Role-Based AI Agents

Jurix models courtroom interactions using specialized agents:

- **ProsecutorAgent**  
  Constructs legally grounded arguments using structured legal datasets.

- **DefenseAgent**  
  Generates counter-arguments, identifies inconsistencies, and builds defense strategies.

- **JudgeAgent**  
  Moderates proceedings, evaluates arguments, and delivers reasoned verdicts.

---

### 📂 Evidence Intelligence Pipeline

- Supports **PDF, DOCX, images, and raw text**
- OCR + NLP-based parsing converts unstructured input into:
  - Structured facts
  - Key entities
  - Argument-ready summaries
- Human-in-the-loop validation:
  - Edit / approve parsed evidence before simulation

---

### ⚖️ Legal Knowledge Integration

- Indian legal frameworks:
  - IPC (Indian Penal Code)
  - CrPC (Criminal Procedure Code)
  - Constitution of India
- Structured datasets enable **context-aware legal reasoning**, not generic LLM responses

---

### 🔄 Simulation Workflow

1. Upload case files and evidence  
2. Parse and structure legal data  
3. Validate extracted evidence  
4. Initiate courtroom simulation  
5. Observe:
   - Arguments (Prosecution vs Defense)
   - Judicial reasoning
   - Final verdict  

---

### 🧩 Multi-Tier AI Architecture

To ensure robustness and offline capability:

- **Primary Layer** → Custom-trained legal models  
- **Secondary Layer** → Open-source LLMs  
- **Fallback Layer** → External APIs (OpenAI / Gemini)

**Benefits:**
- Reliability  
- Cost control  
- Graceful degradation  

---

## 🏗️ System Architecture

### Frontend
- React (Vite)
- Tailwind CSS  
- Component-driven UI for case management and simulation

### Backend
- Flask (Python)
- REST API architecture
- MongoDB / JSON-based storage

### AI Stack
- PyTorch (agent logic)
- OCR + NLP pipelines
- LLM orchestration (local + API fallback)

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/jurix.git
cd jurix
