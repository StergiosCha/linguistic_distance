# Seven-Dimensional Linguistic Distance Analysis

A comprehensive platform for measuring linguistic distance across **7 independent dimensions**, analyzing historical language evolution using multiple computational methods and linguistic databases.

## Features

### **7 Linguistic Distance Dimensions**

1. **Lexical Distance** - Core vocabulary change using Swadesh 100-word lists with Levenshtein distance
2. **Phonological Distance** - Sound change patterns using ASJP (Automated Similarity Judgment Program) transcriptions
3. **Syntactic Distance** - Enhanced Universal Dependencies treebank analysis with 23 dependency relations
4. **Morphological Distance** - POS tag distribution and morphological feature analysis from UD
5. **Typological Distance (WALS)** - World Atlas of Language Structures discrete features
6. **Typological Distance (Grambank)** - 195 binary grammatical features (GB001-GB195)
7. **Cognate Distance** - Historical relatedness using CLDF (Cross-Linguistic Data Formats) cognate judgments

### **11 Historical Language Pairs**

Covers major Indo-European families:
- **Hellenic**: Ancient Greek → Modern Greek (2500 years)
- **Romance**: Latin → Italian/Spanish/French/Romanian (1500 years)
- **Slavic**: Old Church Slavonic → Bulgarian/Russian/Serbian/Czech (1000 years)
- **Germanic**: Gothic → German (1600 years)
- **Indo-Aryan**: Sanskrit → Hindi (1500 years)

### **Advanced Analytics**

- **Correlation Analysis**: Compare dimensions (e.g., lexical vs. syntactic change)
- **Regression Models**: Time vs. distance with R² scores
- **Visualization Suite**: Heatmaps, bar charts, scatter plots, correlation matrices
- **Statistical Validation**: Pearson correlations, p-values, R² scores
- **CSV Export**: Full results with all metrics

### **Optional URIEL Integration**

- **URIEL+ Featural Distance**: Dense linguistic vector embeddings (if urielplus installed)
- Covers phonology, syntax, and phonetics in a unified feature space

## Quick Start

### Prerequisites

- Python 3.8+
- Flask for API backend
- Required databases (included or auto-downloaded):
  - Swadesh lists (lexical)
  - ASJP transcriptions (phonological)
  - Universal Dependencies treebanks (syntactic/morphological)
  - WALS/Grambank data (typological)
  - CLDF cognate data (historical)

### Installation

```bash
# Clone repository
git clone https://github.com/StergiosCha/linguistic-distance.git
cd linguistic-distance

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install URIEL+ for featural distance
pip install urielplus
```

### Running the Application

```bash
# Start Flask server
python main.py

# Or with gunicorn (production):
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

Open your browser to `http://localhost:5000`

## Usage

### 1. Select Language Pairs

Choose from 11 pre-configured historical pairs:
- Ancient Greek → Modern Greek
- Latin → Italian/Spanish/French/Romanian
- Old Church Slavonic → Bulgarian/Russian/Serbian/Czech
- Gothic → German
- Sanskrit → Hindi

### 2. Select Dimensions

Choose which dimensions to analyze (any combination of 7):
- ☑ Lexical Distance
- ☑ Phonological Distance  
- ☑ Syntactic Distance
- ☑ Morphological Distance
- ☑ WALS Typological
- ☑ Grambank Typological
- ☑ Cognate Distance (CLDF)
- ☑ URIEL Featural (optional)

### 3. Run Analysis

Click **Initialize & Analyze** to:
1. Load required data files (Swadesh, ASJP, UD, WALS, Grambank, CLDF)
2. Calculate distances for each dimension
3. Generate statistical correlations
4. Create visualizations

### 4. View Results

- **Distance Matrix**: Heatmap showing all pairs across all dimensions
- **Bar Charts**: Individual dimension comparisons
- **Correlation Analysis**: How dimensions relate to each other
- **Time Regression**: Distance vs. time separation with R² scores
- **Export**: Download full CSV results

## Project Structure

```
linguistic_distance/
├── main.py                          # Flask API + analysis engine
├── index.html                       # Web UI
├── requirements.txt                 # Python dependencies
├── Procfile                        # Deployment config
├── runtime.txt                     # Python version
├── swadesh_*.txt                   # Lexical data (Swadesh lists)
├── *_asjp.txt                      # Phonological data (ASJP)
├── *-ud-train.conllu               # Syntactic/morphological (UD)
├── sane_wals.xlsx                  # Typological (WALS)
├── grambank_sane_format.csv        # Typological (Grambank)
├── languages.csv / forms.csv       # CLDF cognate data
├── cognates.csv                    # CLDF cognate judgments
├── uploads/                        # User-uploaded files
└── results/                        # Analysis outputs
```

## API Endpoints

### Analysis
- `POST /api/initialize` - Initialize analyzers with data
- `POST /api/analyze` - Run 7-dimensional analysis
- `POST /api/visualize` - Generate visualizations
- `POST /api/export` - Export results to CSV

### Data
- `GET /api/language-pairs` - Get available pairs
- `GET /api/dimensions` - Get available dimensions
- `GET /api/check-files` - Verify data file availability

### Debugging
- `GET /api/health` - Health check
- `GET /api/debug-files` - Debug file paths

## Methodology

### Lexical Distance (Swadesh)
```python
distance = 1 - (matching_cognates / total_concepts)
```
Uses Levenshtein distance < 2 as cognate threshold.

### Phonological Distance (ASJP)
```python
distance = average_levenshtein_distance(asjp_transcriptions)
```
Normalized ASJP transcriptions for basic vocabulary.

### Syntactic Distance (UD)
```python
distance = cosine_distance(deprel_frequencies)
```
23 dependency relations: `nsubj`, `obj`, `obl`, `advmod`, `amod`, etc.

### Morphological Distance (UD)
```python
distance = cosine_distance(pos_tag_frequencies)
```
17 UPOS tags: `NOUN`, `VERB`, `ADJ`, etc.

### Typological Distance (WALS)
```python
distance = hamming_distance(wals_features) / total_features
```
Discrete structural features (e.g., word order, case systems).

### Typological Distance (Grambank)
```python
distance = hamming_distance(grambank_features) / 195
```
195 binary grammatical features (GB001-GB195).

### Cognate Distance (CLDF)
```python
distance = 1 - (shared_cognate_sets / shared_concepts)
```
Uses CLDF cognate judgments from historical linguistics.

## Statistical Analysis

### Correlation Matrix
Computes Pearson correlations between all dimension pairs:
```python
correlation, p_value = pearsonr(dimension_1, dimension_2)
```

### Time Regression
Linear regression of distance vs. time separation:
```python
model = LinearRegression().fit(years, distances)
R² = r2_score(distances, predictions)
```

## Data Sources

- **Swadesh Lists**: https://en.wiktionary.org/wiki/Appendix:Swadesh_lists
- **ASJP Database**: https://asjp.clld.org/
- **Universal Dependencies**: https://universaldependencies.org/
- **WALS**: https://wals.info/
- **Grambank**: https://grambank.clld.org/
- **CLDF**: https://cldf.clld.org/

## Deployment

### Local
```bash
python main.py
```

### Production (Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:$PORT main:app
```

### Heroku/Railway
```bash
# Procfile included
web: gunicorn main:app
```

## Performance

- **Initialization**: 5-10 seconds (loads all databases)
- **Analysis**: 2-5 seconds per language pair
- **Visualization**: 1-2 seconds per chart
- **Full 11-pair analysis**: ~30-45 seconds

## Troubleshooting

### Missing URIEL
```bash
pip install urielplus
```
URIEL is optional; other dimensions work without it.

### Missing Data Files
Ensure these files exist in project root:
- `swadesh_*.txt` (lexical)
- `*_asjp.txt` (phonological)  
- `*-ud-train.conllu` (syntactic)
- `sane_wals.xlsx` (WALS)
- `grambank_sane_format.csv` (Grambank)
- `languages.csv`, `forms.csv`, `cognates.csv` (CLDF)

### File Upload Issues
Check `uploads/` folder permissions:
```bash
chmod 755 uploads/
```

## Example Results

### Ancient Greek → Modern Greek
```
Lexical:        0.34 (66% cognate retention)
Phonological:   0.42 (moderate sound change)
Syntactic:      0.28 (relatively stable)
Morphological:  0.31 (case system erosion)
WALS:          0.21 (typologically similar)
Grambank:      0.18 (structurally conservative)
Cognate:       0.25 (high historical continuity)
```

### Latin → French
```
Lexical:        0.51 (49% cognate retention)
Phonological:   0.63 (extensive sound shifts)
Syntactic:      0.45 (SVO vs. SOV reanalysis)
Morphological:  0.58 (case loss, article gain)
WALS:          0.35 (moderate typological shift)
Grambank:      0.29 (grammatical restructuring)
Cognate:       0.38 (clear Romance lineage)
```

## Research Applications

- **Historical Linguistics**: Quantify language change rates
- **Comparative Linguistics**: Multi-dimensional family comparisons
- **Typology**: Correlate structural and historical distances
- **Language Contact**: Detect borrowing vs. inheritance
- **Phylogenetics**: Inform computational phylogeny
- **Sociolinguistics**: Model language divergence timelines

## Contributing

Contributions welcome! Areas for expansion:
1. Additional language families (Semitic, Sino-Tibetan, etc.)
2. More UD treebanks for better coverage
3. Semantic distance (WordNet, conceptual spaces)
4. Pragmatic distance (discourse markers, politeness)
5. Orthographic distance (script evolution)

## License

MIT License - see LICENSE file for details

## Funding

This project is funded by the **European Union** under the **ERC Advanced Grant (ADG)**:

**PhylProGramm** - Grant Agreement No. 101096554

## Contact

**Stergios Chatzikyriakidis**  
Email: stergios.chatzikyriakidis@uoc.gr  
Institution: University of Crete

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{linguistic_distance_2025,
  title = {Seven-Dimensional Linguistic Distance Analysis},
  author = {Chatzikyriakidis, Stergios},
  year = {2025},
  url = {https://github.com/StergiosCha/linguistic-distance}
}
```

## Acknowledgments

- Universal Dependencies project for treebanks
- ASJP database for phonological transcriptions
- WALS and Grambank for typological features
- CLDF initiative for cognate data
- Swadesh list curators
- urielplus library maintainers
