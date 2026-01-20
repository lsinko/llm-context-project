## TL;DR — brzo pokretanje (Windows PowerShell)

```powershell
git clone https://github.com/lsinko/llm-context-project.git
cd llm-context-project

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python src/01_check_kaggle_csv.py
python src/02_clean_kaggle.py
python src/03_fetch_hf_candidates.py
python src/04_integrate.py
python src/05_analyze_visualize.py
python src/06_store_db.py
python src/07_api.py

Nakon pokretanja API-ja testiranje:

http://127.0.0.1:5000/health

http://127.0.0.1:5000/models

http://127.0.0.1:5000/repos

# LLM Projekt — Projektna dokuemntacija 

## 1. Uvod i cilj projekta
Cilj projekta je istražiti utjecaj veličine kontekstnog prozora (context_window) velikih jezičnih modela na njihove performanse i popularnost. Performanse se analiziraju na razini pojedinih redaka iz Kaggle skupa podataka (row-level), dok se popularnost promatra na razini službenih Hugging Face repozitorija (repo-level) putem metrika poput broja preuzimanja i broja “likeova”. Projekt obuhvaća prikupljanje podataka iz heterogenih izvora, njihovu obradu i integraciju, pohranu u bazu podataka, izradu REST API sučelja te bazičnu analizu i vizualizaciju.

## 2. Izvori podataka
Projekt koristi dva heterogena izvora podataka:
1) Kaggle (CSV): “Large Language Models Comparison Dataset” (sadrži tehničke metrike poput context_window, latencije i brzine generiranja).  
2) Hugging Face Hub API (JSON): metapodaci o popularnosti modela (npr. downloads, likes i downloadsAllTime) dohvaćeni kroz API.

Integracija se provodi mapiranjem Kaggle redaka na službene Hugging Face repozitorije. 

## 3. Struktura projekta
Projekt je organiziran u sljedeće direktorije:
- `src/` sadrži Python skripte koje čine pipeline (prikupljanje, obrada, integracija, analiza, pohrana i API).
- `data/raw/` sadrži sirove ulazne podatke (CSV i JSON) dobivene iz izvora.
- `data/processed/` sadrži obrađene i integrirane izlaze (CSV) te SQLite bazu.
- `reports/figures/` sadrži generirane grafove.

## 4. Preduvjeti i instalacija
Projekt se izvodi u Python virtualnom okruženju.

U root direktoriju projekta najprije se pokreće:

```Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

```


Alternativa (bash/WSL/Linux/macOS)
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt



## 5. Pipeline (redoslijed pokretanja skripti)

Skripte se pokreću iz root direktorija projekta i redoslijed je bitan jer svaki korak stvara izlaze koje koriste sljedeći koraci.

### 5.1 Provjera ulaznog Kaggle CSV-a

Skripta provjerava da je Kaggle CSV čitljiv i da sadrži očekivane stupce.

```Windows PowerShell
python src/01_check_kaggle_csv.py
```

### 5.2 Čišćenje Kaggle podataka

Skripta čisti Kaggle CSV, standardizira tipove podataka te priprema stupce za integraciju.

```Windows PowerShell
python src/02_clean_kaggle.py
```
### 5.3 
Dohvat Hugging Face kandidata (API)

Skripta dohvaća relevantne Hugging Face metapodatke i sprema ih u `data/raw/` kao JSON.

```Windows PowerShell
python src/03_fetch_hf_candidates.py
```
### 5.4 Integracija Kaggle + Hugging Face

Skripta integrira Kaggle podatke i Hugging Face metapodatke te generira dva izlaza:

- `data/processed/merged_llm_data.csv` (row-level; jedan redak po Kaggle retku)
- `data/processed/merged_llm_data_repo_level.csv` (repo-level; agregirano po hf_repo_id)

```Windows PowerShell
python src/04_integrate.py
```

### 5.5 Analiza i vizualizacija

Skripta radi bazičnu analizu i generira grafove u `reports/figures/`. Također sprema repo-level normalizirane varijable u:

`data/processed/merged_llm_data_repo_level_normalized.csv`

```Windows PowerShell
python src/05_analyze_visualize.py
```

### 5.6 Pohrana u SQLite bazu

Skripta kreira SQLite bazu te puni tablice podacima iz obrađenih CSV-ova. Baza se sprema kao:

`data/processed/llm_context.db`

```Windows PowerShell
python src/06_store_db.py
```

### 5.7 REST API (Flask)

Skripta pokreće Flask razvojni server i izlaže REST API za dohvat podataka iz SQLite baze podataka.

```Windows PowerShell
python src/07_api.py
```

Server se tada izvodi na:

```
http://127.0.0.1:5000
```

## 6. Testiranje REST API-ja

API se može testirati preko web preglednika ili preko terminala.

### 6.1 Testiranje preko web preglednika

Primjeri:

```
http://127.0.0.1:5000/health
http://127.0.0.1:5000/models
http://127.0.0.1:5000/repos
```

### 6.2 Testiranje preko terminala (Windows PowerShell)

U PowerShellu je curl često alias za Invoke-WebRequest, pa je preporučeno koristiti curl.exe kako bi se dobilo standardno ponašanje curl alata.

```Windows PowerShell
curl.exe "http://127.0.0.1:5000/health"
curl.exe "http://127.0.0.1:5000/models"
curl.exe "http://127.0.0.1:5000/repos"
curl.exe "http://127.0.0.1:5000/models?provider=Meta%20AI&min_context_window=200000"
```

Ako URL sadrži znak `&`, potrebno je cijeli URL staviti u navodnike, inače PowerShell interpretira `&` kao operator.

## 7. Pregled SQLite baze podataka

SQLite baza se nalazi u:

`data/processed/llm_context.db`

Primjeri korisnih SQL upita za provjeru konzistentnosti:

```sql
SELECT COUNT(*) AS n_rows FROM llm_row;
SELECT COUNT(*) AS n_repos FROM llm_repo;

SELECT kaggle_row_id, COUNT(*) AS c
FROM llm_row
GROUP BY kaggle_row_id
HAVING c > 1;

SELECT COUNT(*) AS null_repo
FROM llm_row
WHERE hf_repo_id IS NULL OR TRIM(hf_repo_id) = '';
```

## 8. Izlazni artefakti

Projekt generira sljedeće ključne artefakte:

- Integrirani CSV (row-level): `data/processed/merged_llm_data.csv`
- Integrirani CSV (repo-level): `data/processed/merged_llm_data_repo_level.csv`
- Normalizirani repo-level CSV: `data/processed/merged_llm_data_repo_level_normalized.csv`
- SQLite baza: `data/processed/llm_context.db`
- Grafovi: `reports/figures/*.png`