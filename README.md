# European Parliament Vote Classifier

## Governance Relevance
This project demonstrates applied data analytics for legislative accountability – a core strand of Governance GP's work on transparent and accountable institutions. By predicting political group membership from voting records, it shows how data can be used to measure institutional cohesion, party discipline, and legislative behavior – methods transferable to governance diagnostics, parliamentary strengthening, and public sector accountability work.

## Responsible AI & Limitations
- High feature-to-sample ratio in EU version (2,072 features, 728 MEPs after cleaning) causes mild overfitting despite dropout – noted as future PCA work
- 1984 US data and 2014-2019 EP data are historical – not current governance indicators
- No government stakeholder, research prototype only

A multi-class neural network classifier that predicts which **European Parliament political group** an MEP belongs to based on their voting record across 2,072 key legislative votes during the 8th parliamentary term (2014–2019).

## Dataset

**Source:** [VoteWatch Europe — Professor Simon Hix, EUI](https://simonhix.com/projects/)

- 835 MEPs across 8 political groups
- 10,252 total votes reduced to 2,072 final votes using domain knowledge
- Covers the full 8th term of the European Parliament (2014–2019)

**To run this project, download the EP8 dataset from the link above and place both files in the project folder.**

## Political Groups

| Group | Abbreviation | Political Position |
|---|---|---|
| Group of the European People's Party | EPP | Centre-right |
| Group of the Progressive Alliance of Socialists and Democrats | S&D | Centre-left |
| European Conservatives and Reformists Group | ECR | Right |
| Group of the Alliance of Liberals and Democrats for Europe | ALDE | Centre |
| Group of the Greens/European Free Alliance | Greens/EFA | Left |
| Confederal Group of the European United Left - Nordic Green Left | GUE/NGL | Far-left |
| Europe of Freedom and Direct Democracy Group | EFDD | Eurosceptic |
| Europe of Nations and Freedom Group | ENF | Far-right |

## Project Structure

```
european-parliament-vote-classifier/
├── train.py              # Data loading, cleaning, feature selection, training, saving
├── test.py               # Load saved model and predict on real MEPs
├── ep_model.keras        # Saved trained model
├── EP8_RCVs_2019_06_25.xlsx      # Raw voting data (download separately)
├── EP8_Voted_docs.xlsx           # Vote metadata (download separately)
└── README.md
```

## How It Works

### Feature Selection
The raw dataset contains 10,252 votes. Rather than feeding all of them into the model, only **final votes** are kept — these are the decisive votes on complete legislative texts, filtered using the `Final vote?` column from the vote metadata file. This reduces the feature space to 2,072 meaningful votes and reflects how political scientists approach EP data.

### Data Cleaning
- Vote code `0` (not yet an MEP) replaced with `NaN` then filled as `5` (did not vote)
- Vote code `6` (data error) replaced with `5`
- Non-attached Members excluded — they have no shared political pattern
- MEPs with missing group labels dropped

### Vote Encoding
| Code | Meaning |
|---|---|
| 1 | For |
| 2 | Against |
| 3 | Abstention |
| 4 | Absent |
| 5 | Did not vote |

### Model Architecture
```
Input (2072 features)
→ Dense(64, relu)
→ Dropout(0.2)
→ Dense(32, relu)
→ Dense(8, softmax)
```

- Multi-class classification using `softmax` output (8 political groups)
- Loss: `categorical_crossentropy`
- Optimizer: `Adam(learning_rate=0.0005)`
- Train/test split: 80/20
- EarlyStopping with `patience=10` and `restore_best_weights=True`

### Results
- Validation accuracy: **~88%** across 8 political groups
- Baseline (random guessing): 12.5%

### Known Limitation
The dataset has 2,072 features but only 728 MEPs after cleaning. This high feature-to-sample ratio causes mild overfitting despite dropout regularization. A future improvement would be applying PCA or selecting only the most politically contentious votes to further reduce the feature space.

## How to Run

Install dependencies:
```bash
pip install tensorflow pandas numpy openpyxl
```

Train the model:
```bash
python train.py
```

Test with real MEPs from the dataset:
```bash
python test.py
```

## Example Prediction

```
MEP: Miriam DALLI
Actual group:    Group of the Progressive Alliance of Socialists and Democrats
Predicted group: Group of the Progressive Alliance of Socialists and Democrats
Confidence: 94.3%
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- openpyxl (for reading .xlsx files)

## Data Citation

Simon Hix, Doru Frantescu & Sara Hagemann (2022) *VoteWatch Europe European Parliament and EU Council Voting Data*, September 2022.
