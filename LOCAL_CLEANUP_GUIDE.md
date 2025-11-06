# Guida: Pulizia Directory Locale Groundeep

Questa guida ti aiuta a organizzare la directory locale `/home/student/Desktop/Groundeep` dopo aver separato la pipeline di analisi.

---

## 📂 Struttura Attuale

```
/home/student/Desktop/
├── Groundeep/                    # ← Progetto originale (da pulire)
│   ├── pipeline_refactored/      # ← Codice pipeline con adapters
│   ├── networks/                 # ← Modelli .pkl
│   ├── stimuli_dataset_*/        # ← Dataset
│   ├── src/                      # ← Script di analisi/training
│   ├── results/                  # ← Output analisi
│   └── ...
│
└── groundeep-analysis/           # ← Nuova repo (già pronta per GitHub)
    └── groundeep_analysis/       # ← Solo codice pipeline
```

---

## 🎯 Obiettivo

**Groundeep (locale):**
- Contenere modelli, dati, e script specifici del tuo progetto
- Usare `groundeep-analysis` come libreria installata

**groundeep-analysis (GitHub):**
- Contenere solo la pipeline generica
- Nessun modello o dato specifico

---

## 🧹 Step di Pulizia

### 1. Cosa Tenere in `/home/student/Desktop/Groundeep`

✅ **DA TENERE:**
```
Groundeep/
├── pipeline_refactored/          # ← MANTIENI (versione con adapters)
├── networks/                     # ← MANTIENI (i tuoi modelli)
├── stimuli_dataset_*/            # ← MANTIENI (i tuoi dati)
├── src/                          # ← MANTIENI (script specifici)
├── results/                      # ← MANTIENI (risultati analisi)
├── groundeep/                    # ← MANTIENI (venv)
├── setup.py                      # ← MANTIENI (setup locale)
├── src/configs/                  # ← MANTIENI (config locali)
└── test_adapters.py              # ← MANTIENI (test locale)
```

❌ **DA RIMUOVERE (opzionale):**
```
Groundeep/
├── __pycache__/                  # ← Cache Python
├── .pytest_cache/                # ← Cache pytest
├── *.pyc                         # ← File compilati
├── .ipynb_checkpoints/           # ← Checkpoint Jupyter
└── outputs/                      # ← Output Hydra temporanei
```

### 2. Comandi di Pulizia

```bash
cd /home/student/Desktop/Groundeep

# Rimuovi cache e file temporanei
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Rimuovi output Hydra temporanei (opzionale)
rm -rf outputs/ multirun/

# Rimuovi vecchi risultati se non servono (ATTENZIONE!)
# rm -rf results/  # ← Solo se sei sicuro!
```

### 3. Aggiorna .gitignore locale

Aggiungi al `.gitignore` di Groundeep:

```bash
cat >> .gitignore << 'EOF'
# Python cache
__pycache__/
*.py[cod]
*$py.class
*.pyc
*.pyo

# Jupyter
.ipynb_checkpoints/

# Hydra
outputs/
multirun/

# Virtual env
groundeep/
venv/
env/

# Results (se non vuoi committarli)
results/

# Models (troppo grandi per git)
networks/**/*.pkl
networks/**/*.pth

# Datasets (troppo grandi per git)
stimuli_dataset_*/*.npz
EOF
```

---

## 🔗 Connettere le Due Repo

### Opzione 1: Usa groundeep-analysis come Libreria Installata

```bash
# In Groundeep locale, installa la pipeline da GitHub (dopo il push)
cd /home/student/Desktop/Groundeep
source groundeep/bin/activate
pip install git+https://github.com/francesco-cal98/groundeep-analysis.git

# Ora puoi usarla nel tuo codice
python
>>> from groundeep_analysis.core import ModelManager
>>> # Usa i tuoi modelli locali
```

### Opzione 2: Usa groundeep-analysis in Editable Mode (per sviluppo)

```bash
cd /home/student/Desktop/Groundeep
source groundeep/bin/activate

# Installa la pipeline locale in modalità editable
pip install -e ../groundeep-analysis

# Modifiche a groundeep-analysis saranno subito disponibili
```

### Opzione 3: Mantieni Copia Locale di pipeline_refactored

```bash
# Se preferisci, mantieni pipeline_refactored/ in Groundeep
# e continua a usarla come prima

# In questo caso, puoi sincronizzare le modifiche:
cd /home/student/Desktop/Groundeep

# Copia modifiche da groundeep-analysis quando serve
cp -r ../groundeep-analysis/groundeep_analysis/* pipeline_refactored/
```

---

## 📦 Struttura Finale Consigliata

```
/home/student/Desktop/
│
├── Groundeep/                           # ← Progetto di ricerca
│   ├── .git/                            # ← Git locale o GitHub privato
│   ├── .gitignore                       # ← Ignora cache, modelli, dati
│   ├── pipeline_refactored/             # ← Copia locale pipeline (sync con groundeep-analysis)
│   ├── networks/                        # ← I tuoi modelli (non su git)
│   ├── stimuli_dataset_*/               # ← I tuoi dati (non su git)
│   ├── src/                             # ← Script specifici
│   │   ├── configs/                     # ← Config locali
│   │   └── main_scripts/                # ← Script di analisi/training
│   ├── results/                         # ← Output (non su git)
│   ├── groundeep/                       # ← Venv (non su git)
│   ├── setup.py                         # ← Setup locale
│   └── README.md                        # ← Descrizione progetto
│
└── groundeep-analysis/                  # ← Pipeline generica (su GitHub pubblico)
    ├── .git/                            # ← Git pubblico
    ├── groundeep_analysis/              # ← Solo codice pipeline
    ├── examples/                        # ← Esempi generici
    ├── tests/                           # ← Test (da aggiungere)
    ├── setup.py                         # ← Setup pip
    ├── requirements.txt
    ├── LICENSE
    └── README.md                        # ← Documentazione pubblica
```

---

## 🚦 Workflow Consigliato

### Per Sviluppo Pipeline
```bash
# 1. Lavora su groundeep-analysis
cd /home/student/Desktop/groundeep-analysis
# ... fai modifiche ...

# 2. Testa localmente
cd /home/student/Desktop/Groundeep
source groundeep/bin/activate
python src/main_scripts/analyze_modular.py

# 3. Se funziona, pusha su GitHub
cd /home/student/Desktop/groundeep-analysis
git add .
git commit -m "Feature: ..."
git push
```

### Per Analisi Dati
```bash
# 1. Lavora in Groundeep
cd /home/student/Desktop/Groundeep
source groundeep/bin/activate

# 2. Usa pipeline (da pip o locale)
python src/main_scripts/analyze_modular.py

# 3. Risultati salvati in results/
```

---

## 📝 Checklist Finale

Prima di pushare groundeep-analysis su GitHub:

- [x] Informazioni personali aggiornate (nome, email, GitHub username)
- [x] Nessun file `.pkl` (modelli) nella repo
- [x] Nessun file `.npz` (dati) nella repo
- [x] Nessun `results/` o output specifici
- [x] Solo codice generico della pipeline
- [ ] Test funzionanti (opzionale: aggiungi test con dati toy)
- [ ] README completo e professionale
- [x] LICENSE presente
- [x] .gitignore configurato correttamente

Prima di committare Groundeep su git (se vuoi):

- [ ] .gitignore esclude modelli, dati, results
- [ ] Solo codice e configurazioni committati
- [ ] README descrive il progetto di ricerca
- [ ] Repository privato (se contiene dati sensibili)

---

## 🆘 In Caso di Dubbi

### "Ho modificato qualcosa in pipeline_refactored, come sincronizzo?"

```bash
# Copia modifiche da Groundeep a groundeep-analysis
cd /home/student/Desktop/Groundeep
cp -r pipeline_refactored/core/adapters/* \
      ../groundeep-analysis/groundeep_analysis/core/adapters/

# Poi pusha le modifiche
cd ../groundeep-analysis
git add .
git commit -m "Sync: updates from local development"
git push
```

### "Voglio usare sempre l'ultima versione di groundeep-analysis"

```bash
# Installa in editable mode
cd /home/student/Desktop/Groundeep
source groundeep/bin/activate
pip install -e ../groundeep-analysis

# Ora modifiche a groundeep-analysis sono immediate
```

### "Ho committato per errore un file .pkl grande"

```bash
# Rimuovi dal commit (PRIMA di pushare)
git rm --cached networks/model.pkl
git commit --amend -m "Remove large model file"

# Aggiorna .gitignore
echo "networks/**/*.pkl" >> .gitignore
```

---

**Ultima modifica:** Gennaio 2025
**Autore:** Francesco Maria Calistroni
