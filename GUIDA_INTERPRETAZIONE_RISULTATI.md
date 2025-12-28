# Guida all'Interpretazione dei Risultati dell'Analisi iDBN

## Indice
1. [Panoramica della Struttura](#panoramica-della-struttura)
2. [Power-Law Fitting (powerfit_pairs)](#power-law-fitting)
3. [Probes](#probes)
4. [Dimensionality Analysis](#dimensionality-analysis)
5. [Behavioral Analysis](#behavioral-analysis)
6. [Feature Analysis](#feature-analysis)
7. [Reconstruction Analysis](#reconstruction-analysis)
8. [Label Histograms](#label-histograms)

---

## Panoramica della Struttura

I risultati per ogni architettura (es. `iDBN_1500_1500_multimodal`) sono organizzati in diverse cartelle, ognuna contenente analisi specifiche sulle rappresentazioni neurali della numerosità:

```
iDBN_1500_1500_multimodal/
├── powerfit_pairs/          # Analisi power-law delle distanze
├── probes/                  # Linear probes per decodifica
├── dimensionality/          # Analisi geometrica delle rappresentazioni
├── behavioral/              # Analisi comportamentale (estimation/comparison)
├── feature_analysis/        # Correlazioni tra feature visive
├── reconstruction/          # Qualità della ricostruzione
└── label_histograms/        # Distribuzione delle numerosità
```

---

## 1. Power-Law Fitting

**Cartella:** `powerfit_pairs/`

### Cosa misura
Analizza se le rappresentazioni neurali seguono una **power-law** (legge di potenza), caratteristica tipica dell'Approximate Number System (ANS) biologico. Secondo Weber's Law, la discriminabilità tra numerosità diminuisce con il rapporto n₁/n₂.

### File prodotti

#### `pairs_table_*.csv`
Tabella con tutte le coppie di numerosità analizzate.

**Colonne:**
- `i, j`: Indici delle numerosità confrontate (es. 1→numerosità 1, 2→numerosità 2)
- `deltaN`: Differenza assoluta tra le numerosità (j - i)
- `distance`: Distanza euclidea nello spazio delle rappresentazioni neurali tra i centroidi delle due numerosità

**Interpretazione:**
- Distanze più piccole → le numerosità sono rappresentate in modo più simile
- Distanze più grandi → le numerosità sono ben separate nello spazio rappresentazionale
- La crescita delle distanze dovrebbe rallentare per numerosità grandi (se segue Weber's Law)

**Esempio:**
```
i,j,deltaN,distance
1,2,1,2.245    # Numerosità 1 vs 2: distanza 2.245
5,6,1,0.923    # Numerosità 5 vs 6: distanza 0.923 (più vicine in proporzione!)
```

#### `params_*.csv`
Parametri del fit della power-law: **distance = a × (numerosity)^b**

**Colonne:**
- `a`: Coefficiente di scala
- `b`: Esponente della power-law
- `r2`: Coefficiente di determinazione (bontà del fit, 0-1)
- `n_points`: Numero di coppie usate per il fit

**Interpretazione dell'esponente `b`:**
- `b ≈ 0.5-0.7`: Power-law tipica dell'ANS biologico (comportamento logaritmico-like)
- `b ≈ 1.0`: Relazione lineare (spacing uniforme)
- `b < 0.5`: Compressione forte (numerosità grandi molto vicine)
- `b > 1.0`: Relazione super-lineare (rara)

**Interpretazione di `r²`:**
- `r² > 0.9`: Eccellente fit alla power-law
- `0.7 < r² < 0.9`: Buon fit
- `r² < 0.7`: Il modello potrebbe non seguire una power-law chiara

#### `fit_linear_*.png`
Plot in scala **lineare** delle distanze vs. numerosità.

**Interpretazione:**
- **Curva concava** (rallentamento progressivo): Indica compressione logaritmica, tipica dell'ANS
- **Linea retta**: Spacing uniforme delle rappresentazioni
- **Rumore/scatter**: Rappresentazioni inconsistenti

#### `fit_loglog_*.png`
Plot in scala **log-log** delle distanze vs. numerosità.

**Interpretazione:**
- In scala log-log, una power-law pura appare come una **linea retta**
- La **pendenza** della retta corrisponde all'esponente `b`
- Deviazioni dalla linearità indicano che la relazione non è perfettamente una power-law
- Questo plot è più diagnostico per verificare l'effettiva presenza di una power-law

#### `residuals_summary.csv`
Statistiche sui residui del fit (differenza tra distanze osservate e predette).

**Metriche tipiche:**
- Mean residuals: Dovrebbe essere vicino a 0
- Std residuals: Più basso = miglior fit
- Max/min residuals: Identificano outlier

---

## 2. Probes

**Cartella:** `probes/top/`

### Cosa misura
Valuta quanto le **proprietà visive** (anziché la numerosità astratta) siano decodificabili dalle rappresentazioni neurali usando **linear probes** (classificatori lineari semplici).

Se i probes hanno alta accuratezza, significa che la rete codifica ancora informazioni visive di basso livello, non solo numerosità astratta.

### Proprietà testate

#### `labels` (Numero di oggetti)
Decodifica direttamente quanti oggetti ci sono nell'immagine.

#### `cum_area` (Area cumulativa)
Area totale occupata dagli oggetti. Alta correlazione con numerosità quando gli oggetti hanno dimensione costante.

#### `convex_hull` (Convex hull)
Area del poligono convesso che contiene tutti gli oggetti. Misura lo "sparpagliamento" spaziale.

#### `density` (Densità)
Numero di oggetti per unità di area. Correlata con numerosità e area.

#### `mean_item_size` (Dimensione media degli oggetti)
Dimensione media di ciascun oggetto nell'immagine. Permette di verificare se la rete usa la dimensione degli oggetti come euristica per stimare la numerosità.

### File prodotti

#### `probe_summary.csv`
Accuratezze dei linear probes per ciascuna proprietà.

**Esempio:**
```
metric,accuracy
top/cum_area,0.236
top/convex_hull,0.301
top/labels,0.280
top/density,0.255
top/mean_item_size,0.189
```

**Interpretazione:**
- **Accuratezza alta (>0.5)**: La rete codifica fortemente quella proprietà visiva
  - Problema: La rete potrebbe "barare" usando area/densità invece di numerosità
- **Accuratezza bassa (~chance level)**: La rete ha sviluppato rappresentazioni astratte
  - Chance level dipende dal numero di classi (es. ~0.03 per 32 numerosità)
  - ~0.25-0.30 indica ancora dipendenza moderata dalle feature visive
- **Convex hull > altri**: La rete usa informazioni spaziali (distribuzione degli oggetti)
- **Cum_area bassa**: Buono! La rete non usa solo la "quantità di pixel"
- **Mean_item_size bassa**: Ottimo! La rete non si basa sulla dimensione degli oggetti

#### `probe_summary.png`
Barplot delle accuratezze per confronto visivo.

#### `*_confusion.png`
Matrici di confusione per ogni proprietà.

**Come leggerle:**
- **Diagonale forte**: Il probe riesce a decodificare bene quella proprietà
- **Errori sistematici**: Pattern di confusione
  - Errori vicini alla diagonale: Confusione tra valori simili (normale)
  - Bande orizzontali/verticali: Bias verso specifici valori

#### `probe_*_confusion_epoch0.csv`
Dati numerici delle matrici di confusione.

---

## 3. Dimensionality Analysis

**Cartella:** `dimensionality/layer{N}/`

### Cosa misura
Analizza la **geometria** delle rappresentazioni nello spazio neurale ad alta dimensione.

### Sottocartelle

#### `pca_report/`
Analisi PCA (Principal Component Analysis) standard.

##### File chiave:

**`explained_variance.csv`**
Varianza spiegata da ciascuna componente principale.

```
component,variance_ratio
1,0.5362    # 53.62% della varianza totale
2,0.0289    # 2.89%
3,0.0200
...
```

**Interpretazione:**
- **Prima componente dominante (>50%)**: Esiste una direzione principale di variazione
  - Potrebbe essere la "linea numerica" mentale
- **Poche componenti spiegano molta varianza**: Rappresentazioni low-dimensional
  - Esempio: Prime 5 componenti spiegano >60% → compressione efficace
- **Varianza distribuita**: Rappresentazioni ad alta dimensionalità intrinseca

**`pca_projection.png` / `pca_projection_3d.png`**
Proiezioni delle rappresentazioni nelle prime 2-3 componenti principali.

**Cosa cercare:**
- **Ordinamento**: I punti sono ordinati lungo PC1 dalla numerosità 1 alla 32?
  - Sì: Codifica monotona della numerosità
  - No: Codifica più complessa o non lineare
- **Clustering**: Numerosità simili formano cluster separati?
- **Curvatura**: La "linea" è dritta o curva?
  - Curva: Possibile compressione logaritmica
  - Dritta: Codifica lineare
- **Overlap**: Le nuvole di punti si sovrappongono?
  - Overlap alto: Scarsa discriminabilità
  - Separazione netta: Buona rappresentazione

**`tsne_projection.png`**
Proiezione t-SNE (preserva distanze locali).

**Interpretazione:**
- Meno affidabile per distanze globali, ma mostra bene cluster locali
- **Cluster separati per numerosità**: Buon segnale
- **Continuum**: Transizione graduale tra numerosità

**`raw_space_projection.png`**
Proiezione delle rappresentazioni originali (senza PCA).

**`dataset_overview.png`**
Panoramica della distribuzione delle numerosità nel dataset.

##### Sottocartella `feature_colored/`

**NUOVO!** Questa cartella contiene plot PCA separati, ognuno colorato per una diversa feature visiva.

**File generati:**
- `pca_colored_by_labels.png` / `pca_3d_colored_by_labels.png`: PCA colorato per numerosità
- `pca_colored_by_cum_area.png` / `pca_3d_colored_by_cum_area.png`: PCA colorato per area cumulativa
- `pca_colored_by_convex_hull.png` / `pca_3d_colored_by_convex_hull.png`: PCA colorato per convex hull
- `pca_colored_by_density.png` / `pca_3d_colored_by_density.png`: PCA colorato per densità
- `pca_colored_by_mean_item_size.png` / `pca_3d_colored_by_mean_item_size.png`: PCA colorato per dimensione media

**Come interpretarli:**

Questi plot mostrano la stessa proiezione PCA delle rappresentazioni neurali, ma colorati secondo diverse proprietà visive invece che per numerosità. Sono fondamentali per capire **quali feature visive sono codificate nello spazio rappresentazionale**.

**Cosa cercare:**

1. **Gradiente ordinato lungo PC1:**
   - Se vedi un gradiente di colore ordinato lungo la prima componente principale (PC1), significa che quella feature è codificata nella direzione principale di variazione
   - Esempio: Se `pca_colored_by_labels.png` mostra un gradiente chiaro lungo PC1, la numerosità è la dimensione principale
   - Se invece `pca_colored_by_cum_area.png` mostra un gradiente, la rete potrebbe usare l'area totale come proxy per numerosità

2. **Confronto tra features:**
   - Confronta i plot tra loro per capire quale feature domina
   - **Ideale:** Solo `labels` mostra gradiente chiaro → rappresentazione astratta della numerosità
   - **Problematico:** Anche `cum_area` o `density` mostrano gradienti → la rete usa euristiche visive

3. **Separazione vs. Mescolamento:**
   - **Colori separati in cluster:** La feature ha pattern discreti nello spazio
   - **Colori mescolati:** La feature non è ben codificata in modo sistematico

4. **Correlazioni spurie:**
   - Se `mean_item_size` mostra un gradiente inverso rispetto a `labels`, può indicare che il dataset ha correlazione negativa (più oggetti → oggetti più piccoli)
   - Questo aiuta a capire se la rete potrebbe "barare" usando la dimensione invece del numero

**Esempio di analisi:**

```
Plot 'pca_colored_by_labels.png':
→ Gradiente blu→giallo lungo PC1 (numerosità basse→alte)
→ BUONO: La numerosità è la dimensione principale

Plot 'pca_colored_by_cum_area.png':
→ Gradiente simile a labels
→ ATTENZIONE: La rete potrebbe confondere area e numerosità

Plot 'pca_colored_by_mean_item_size.png':
→ Colori mescolati, nessun pattern chiaro
→ OTTIMO: La rete ignora la dimensione degli oggetti
```

**Vantaggi rispetto ai probes:**
- I probes misurano quanto **decodificabile** è una feature (con un modello lineare)
- Questi plot mostrano **dove** e **come** le feature sono organizzate nello spazio
- Complementari: accuracy probe bassa MA gradiente visibile → feature codificata in modo non lineare

#### `pca_geometry/`
Analisi geometrica avanzata delle rappresentazioni.

##### `projection_*_dim2.png` / `projection_*_dim3.png`
Proiezioni in 2D e 3D con analisi della geometria.

**Cosa mostrano (di solito):**
- Centroidi per ciascuna numerosità
- Traiettoria dei centroidi (la "linea numerica")
- Eventuale curvatura

##### `curvature_centroids_*.png`
Misura la **curvatura** della traiettoria dei centroidi nel tempo.

**Interpretazione:**
- **Curvatura alta inizialmente, poi diminuisce**: Compressione logaritmica
- **Curvatura costante**: Curva costante (es. cerchio/spirale)
- **Curvatura ~0**: Linea retta (codifica lineare)

##### `angles_*.png`
Angoli tra vettori successivi nella traiettoria dei centroidi.

**Interpretazione:**
- **Angoli costanti**: Traiettoria regolare
- **Variazioni negli angoli**: Cambio di direzione
  - Può indicare "regimi" diversi (es. subitizing range vs estimation range)

##### `balanced_corr_*.png`
Correlazione tra distanze nello spazio neurale e distanze numeriche.

**Interpretazione:**
- **Correlazione alta (>0.8)**: Le distanze neurali riflettono fedelmente le differenze numeriche
- **Correlazione moderata**: Relazione non lineare
- **Correlazione bassa**: Codifica rumorosa o non monotona

##### Sottocartelle `within_class/` e `between_class/`

**Within-class**: Analizza la variabilità **all'interno** di ciascuna numerosità
- Quanto sono compatte le rappresentazioni per la stessa numerosità?

**Between-class**: Analizza le differenze **tra** numerosità diverse
- Quanto sono separati i cluster?

**`eigenvalue_ratios_*.png`**
Rapporti tra autovalori della matrice di covarianza.

**Interpretazione within-class:**
- Rapporti alti: Rappresentazioni molto variabili in alcune direzioni (anisotropiche)
- Rapporti bassi (~1): Rappresentazioni isotropiche (varianza uniforme)

**Interpretazione between-class:**
- Indica quanto la variabilità tra classi domina su quella within-class
- Idealmente: between-class variance >> within-class variance

---

## 4. Behavioral Analysis

**Cartella:** `behavioral/`

### Sottocartelle

#### `estimation/`
Simulazione di compiti di **stima della numerosità**.

**Struttura:** `estimation/{dataset_type}/{model_type}/`

##### `SGD_regression_confusion.png`
Matrice di confusione per un modello di regressione SGD addestrato a predire la numerosità dalle rappresentazioni.

**Interpretazione:**
- **Diagonale forte**: Stima accurata
- **Bande intorno alla diagonale**: Errori piccoli (normale)
- **Errori lontani dalla diagonale**: Stima molto inaccurata
- **Pattern sistematici**:
  - Sottostima (sotto la diagonale): Il modello predice numerosità minori
  - Sovrastima (sopra la diagonale): Il modello predice numerosità maggiori
  - Per numerosità grandi, spesso si osserva maggiore variabilità (Weber's Law)

#### `fixed_reference/`
Simulazione di compiti di **comparazione** con riferimento fisso.

**Esempio:** `ref8/` = confronto con numerosità di riferimento 8

**Task:** "Questo stimolo ha più o meno di 8 oggetti?"

**Analisi tipiche:**
- Curve psicometriche (probabilità di risposta "maggiore" vs numerosità)
- PSE (Point of Subjective Equality): Numerosità percepita come uguale al riferimento
- JND (Just Noticeable Difference): Soglia di discriminazione

---

## 5. Feature Analysis

**Cartella:** `feature_analysis/`

### `feature_correlations_*.png`
Matrice di correlazione tra feature visive calcolate sugli stimoli.

**Feature tipiche:**
- `labels`: Numero di oggetti
- `cum_area`: Area totale
- `convex_hull`: Area del convex hull
- `density`: Densità
- Altre feature geometriche

**Interpretazione:**
- **Correlazioni alte (>0.7)**: Le feature sono ridondanti
  - Es. cum_area e labels correlati → oggetti di dimensione simile
- **Correlazioni basse**: Le feature sono ortogonali
  - Importante per capire se il dataset controlla adeguatamente le variabili confondenti
- **Pattern attesi**:
  - In dataset "uniform": Bassa correlazione tra numerosità e feature visive
  - In dataset "zipfian": Alta correlazione (distribuzione naturale)

---

## 6. Reconstruction Analysis

**Cartella:** `reconstruction/layer{N}/`

### Cosa misura
Qualità della **ricostruzione** degli stimoli dalle rappresentazioni neurali (per architetture generative come DBN).

### Sottocartelle per metrica

#### `mse/` (Mean Squared Error)
Errore quadratico medio pixel-per-pixel.

**Interpretazione:**
- **MSE basso**: Ricostruzione fedele
- **MSE alto**: Perdita di informazione
- **MSE vs numerosità**: Pattern sistematici?
  - MSE costante: La rete ricostruisce ugualmente bene tutte le numerosità
  - MSE cresce: Numerosità grandi più difficili da ricostruire

#### `ssim/` (Structural Similarity Index)
Misura la similarità percettiva (non solo pixel-level).

**Interpretazione:**
- **SSIM ~ 1**: Ricostruzione percettivamente identica
- **SSIM < 0.5**: Ricostruzione molto degradata
- Generalmente più informativo di MSE per immagini

#### `afp/` (Adaptive Feature Pooling)
Metrica specifica per valutare la preservazione di feature visive.

### File prodotti

##### `{metric}_heatmap_cumarea.png` / `{metric}_heatmap_hull.png`
Heatmap della metrica in funzione di numerosità e proprietà visiva (cumarea o hull).

**Interpretazione:**
- **Bande verticali**: L'errore dipende principalmente dalla numerosità
- **Bande orizzontali**: L'errore dipende dalla proprietà visiva
- **Pattern diagonali**: Interazione tra numerosità e proprietà visiva

##### `{metric}_vs_numerosity_cumarea.png`
Plot dell'errore vs numerosità, colorato per cumarea.

**Cosa cercare:**
- **Trend crescente**: Numerosità alte più difficili da ricostruire
- **Scatter alto**: Variabilità dipendente da altre feature
- **Separazione per colore**: L'errore dipende anche da cumarea

---

## 7. Label Histograms

**Cartella:** `label_histograms/`

### `{dataset_type}.png`
Istogramma della distribuzione delle numerosità nel dataset.

**Interpretazione:**
- **Uniform**: Tutte le numerosità hanno uguale frequenza (istogramma piatto)
  - Usato per testare rappresentazioni non biased
- **Zipfian**: Numerosità piccole molto più frequenti (distribuzione naturale)
  - Simula distribuzione ecologica
  - Può indurre bias verso numerosità piccole

---

## Come Leggere i Risultati Complessivamente

### 1. Inizia con Power-Law Fitting
- Controlla l'esponente `b` e `r²`
- Guarda i plot linear e loglog
- **Domanda:** La rete ha sviluppato una rappresentazione ANS-like?

### 2. Analizza la Geometria (Dimensionality)
- Guarda le proiezioni PCA
- Controlla la varianza spiegata
- Analizza curvatura e correlazioni
- **Domanda:** Come è organizzata la "linea numerica" nel cervello della rete?

### 3. Verifica l'Astrazione (Probes)
- Controlla le accuratezze dei probes
- Guarda le matrici di confusione
- **Domanda:** La rete usa ancora feature visive o ha astratto la numerosità?

### 4. Valuta il Comportamento (Behavioral)
- Analizza le performance di stima
- Controlla le curve di comparazione
- **Domanda:** La rete si comporta come un ANS biologico?

### 5. Contestualizza (Feature Analysis + Reconstruction)
- Verifica le correlazioni nel dataset
- Valuta la qualità della ricostruzione
- **Domanda:** Cosa controlla il dataset? Cosa perde la rete nella codifica?

---

## Metriche Chiave da Riportare

### Per un paper/presentazione, focus su:

1. **Power-law exponent (b)** e **r²**
   - Indicatore principale di ANS-like behavior

2. **Probe accuracies** (incluso mean_item_size!)
   - Quanto la rete dipende da feature visive?
   - Mean_item_size basso indica indipendenza dalla dimensione degli oggetti

3. **PCA variance ratio (PC1)**
   - Dimensionalità intrinseca delle rappresentazioni

4. **PCA feature-colored plots** (NUOVO!)
   - Visualizzazione qualitativa di quali feature dominano lo spazio rappresentazionale
   - Confronto visivo tra numerosità e altre proprietà visive

5. **Curvature metrics**
   - Evidenza di compressione logaritmica

6. **Behavioral performance**
   - Accuratezza di stima/comparazione
   - Analogia con dati comportamentali umani

---

## Novità Recenti

### 1. Probing per `mean_item_size`
Aggiunto un nuovo linear probe che testa se la rete decodifica la dimensione media degli oggetti. Questo è cruciale per verificare che la rete non usi la dimensione come scorciatoia per stimare la numerosità.

**Dove trovarlo:**
- `probes/top/probe_summary.csv`: Include ora `top/mean_item_size`
- `probes/top/top_mean_item_size_confusion.png`: Matrice di confusione

### 2. Plot PCA colorati per feature
Per ogni layer analizzato, vengono ora generati plot PCA separati colorati per ogni feature visiva disponibile.

**Dove trovarli:**
- `dimensionality/layer{N}/pca_report/feature_colored/`
- File: `pca_colored_by_{feature}.png` e `pca_3d_colored_by_{feature}.png`

**Features visualizzate:**
- labels (numerosità)
- cum_area (area cumulativa)
- convex_hull (convex hull)
- density (densità)
- mean_item_size (dimensione media) - NUOVO!

**Utilizzo consigliato:**
1. Apri tutti i plot affiancati
2. Confronta visivamente quale feature mostra il gradiente più chiaro lungo PC1
3. Se solo `labels` mostra ordinamento chiaro → ottimo! Rappresentazione astratta
4. Se altre feature mostrano pattern simili → attenzione, possibili confound

---

## Conclusioni

Questa pipeline produce un'analisi multidimensionale delle rappresentazioni neurali della numerosità, permettendo di valutare:

- **Similarità con l'ANS biologico** (power-law, logarithmic compression)
- **Grado di astrazione** (indipendenza da feature visive, inclusa dimensione oggetti)
- **Geometria rappresentazionale** (struttura della "linea numerica")
- **Comportamento emergente** (performance in task cognitivi)
- **Organizzazione spaziale delle features** (plot PCA colorati)

Confronta sempre risultati tra:
- Diverse architetture (es. iDBN_1500_500 vs iDBN_1500_1500_multimodal)
- Diversi layer (layer2 vs layer3)
- Diversi dataset (uniform vs zipfian)

Questo ti permette di isolare l'effetto di architettura, profondità, e distribuzione di training.
