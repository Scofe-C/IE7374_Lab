# Data Labeling Labs

A three-part lab series on **weak supervision** using [Snorkel](https://snorkel.ai/).
The core idea: instead of manually labeling training data, we write programmatic heuristics
that generate noisy labels automatically, then let Snorkel's `LabelModel` combine them intelligently.

**Domain**: phishing email detection — classifying emails as `PHISHING` or `LEGITIMATE`.
All labs share the same dataset (`lab1/data/emails.csv`, 80 emails, no downloads required).

---

## Lab Overview

| Lab | Notebook | Key Concept |
|-----|----------|-------------|
| `lab1/` | `01_email_labeling_functions.ipynb` | Write 10 heuristic LFs (keyword, regex, structural) that vote on each email; combine noisy votes with `LabelModel`; train a downstream `LogisticRegression` |
| `lab2/` | `02_email_data_augmentation.ipynb` | Write 6 transformation functions to synthetically expand the labeled training set; compare model performance before and after augmentation |
| `lab3/` | `03_email_data_slicing.ipynb` | Write 7 slicing functions to identify challenging subgroups; monitor per-slice F1 with `Scorer`; improve weak slices via targeted oversampling |

---

## Folder Structure

```
data_labeling_labs/
├── README.md                          ← this file
├── requirements.txt                   ← install once for all labs
├── lab1/
│   ├── emails.csv                 ← shared dataset (used by all labs)
│   └──01_email_labeling_functions.ipynb
│   
├── lab2/
│   └──02_email_data_augmentation.ipynb
│    
└── lab3/
    └── 03_email_data_slicing.ipynb  
```

---

## Setup (PyCharm / Windows)

### 1. Create a virtual environment

Open the PyCharm terminal (`Alt+F12`) and run:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `snorkel` pulls in several heavy dependencies including PyTorch.
> First install takes 5–10 minutes. Make sure you have at least 3 GB free.

### 3. Configure the interpreter in PyCharm

Go to **File → Settings → Python Interpreter**, click the gear icon, choose
**Add**, select **Existing environment**, and point it at `.venv\Scripts\python.exe`.

### 4. Open notebooks

Open any `.ipynb` file in PyCharm. Select the venv kernel from the top-right
dropdown if it is not already selected. Run all cells top to bottom with
**Run All** or `Shift+Enter` cell by cell.

> Run the labs in order — lab2 and lab3 load the dataset from `../lab1/data/emails.csv`.
