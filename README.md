# PSLM-2020 Poverty Prediction

This project analyzes the Pakistan Social and Living Standards Measurement (PSLM) 2019-2020 dataset to predict poverty levels using income, housing, assets, and other socio-economic indicators. It follows the World Bank's poverty definition ($2.15/day, 2017 PPP) to classify households as "poor" or "non-poor" and employs machine learning for predictive modeling.

## Project Overview

The PSLM-2019/20 dataset, sourced from the Pakistan Bureau of Statistics (PBS), provides comprehensive data on household characteristics across Pakistan. This project processes the dataset to:
- Calculate household income and classify poverty status (Part 1).
- Extract features from multiple dataset sections and train a Random Forest model for poverty prediction (Part 2).

The project includes Jupyter notebooks for data processing and reports summarizing the methodology and findings.

## Prerequisites

- **Python 3.8+**: Ensure Python is installed.
- **Dependencies**: Install required libraries:
  ```bash
  pip install pandas numpy seaborn matplotlib scikit-learn
  ```
- **Dataset**: Download PSLM-2019/20 microdata (.dta files) from the [PBS website](https://www.pbs.gov.pk/content/microdata). Key files include:
  - `sece.dta` (employment)
  - `secf1.dta` (housing)
  - `secf2.dta` (WASH)
  - `secf3.dta` (solid waste)
  - `secg.dta` (assets)
  - `sech.dta` (durables)
  - `secc1.dta` (education)
- **Jupyter Notebook**: For running `.ipynb` files.
- **Word Processor**: For viewing `.docx` reports (e.g., Microsoft Word, LibreOffice).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pslm-poverty-prediction.git
   cd pslm-poverty-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Create `requirements.txt` with: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`.)
3. Place PSLM `.dta` files in the project directory or update file paths in the notebooks.

## Project Structure

- **`Income_Analysis_and_Poverty_Classification.ipynb`**: Processes employment data (Section E) to calculate total annual income and classify households as "poor" or "non-poor" using the World Bank threshold ($784.75/year). Includes a bar chart visualizing poverty distribution.
- **`Multi_Section_Preprocessing.ipynb`**: Loads and preprocesses multiple dataset sections (employment, housing, WASH, assets, etc.) for advanced analysis. Sets up data for machine learning.
- **`Poverty_Classification_Report.docx`**: Details data exploration, income calculation, and poverty classification methodology.
- **`Feature_Engineering_and_Modeling_Report.docx`**: Describes feature extraction (e.g., asset score, housing quality) and Random Forest model training (69.67% accuracy).
- **`.gitignore`**: Ignores Jupyter checkpoints, Python cache, and large `.dta` files.
- **`requirements.txt`**: Lists Python dependencies.

## Usage

1. **Run Income Analysis**:
   - Open `Income_Analysis_and_Poverty_Classification.ipynb` in Jupyter.
   - Ensure `sece.dta` is available.
   - Execute cells to compute incomes and generate poverty labels.
   - View the poverty distribution bar chart.

2. **Run Multi-Section Preprocessing and Modeling**:
   - Open `Multi_Section_Preprocessing.ipynb`.
   - Ensure all required `.dta` files are available.
   - Complete preprocessing and feature engineering (merge datasets, derive features).
   - Train the Random Forest model and evaluate performance (accuracy, confusion matrix).

3. **Review Reports**:
   - Read `Poverty_Classification_Report.docx` for basic analysis details.
   - Read `Feature_Engineering_and_Modeling_Report.docx` for advanced modeling insights.

## Methodology

### Part 1: Income-Based Poverty Classification
- **Data**: Employment data (`sece.dta`).
- **Preprocessing**: Filled missing income values with 0, renamed columns for clarity (e.g., `seaq07` to `report_period`).
- **Feature Engineering**: Calculated annual income from main and secondary occupations; derived total income.
- **Poverty Threshold**: Applied World Bank's $2.15/day (~$784.75/year) to label households.
- **Visualization**: Bar chart of "poor" vs. "non-poor" households.

### Part 2: Feature Extraction and Machine Learning
- **Data**: Merged employment, housing, WASH, assets, durables, and education sections.
- **Features**: Total income, asset score, housing quality index, sanitation score, land ownership, education level, etc.
- **Preprocessing**: Standardized `hhcode`, normalized categorical variables, selected relevant columns.
- **Model**: Random Forest Classifier (80/20 train-test split, `StandardScaler`, `random_state=42`).
- **Performance**: Achieved 69.67% accuracy; analyzed precision, recall, and confusion matrix.

## Notes
- **Dataset Access**: PSLM-2019/20 `.dta` files are not included due to size and licensing. Download from [PBS](https://www.pbs.gov.pk/content/microdata).
- **Visualization**: Bar charts and confusion matrix heatmaps are referenced in reports; implement in notebooks if needed.
- **Extensibility**: Add more features or try other models (e.g., XGBoost, neural networks) for improved accuracy.
