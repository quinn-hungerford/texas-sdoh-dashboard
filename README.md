# Texas Predictive Health (SDOH) Dashboard: Linking Social Determinants to Adult Health Outcomes (2010–2020)

## Project Overview  
This project was developed as the **final project for CS 329E – Elements of Data Visualization (UT Austin, Spring 2025)**.  
Our team developed an **interactive dashboard** in Python (Dash, Plotly, Pandas) that enables users to explore how Social Determinants of Health (SDOH) shape **adult health (via an Adult Health Score)** across Texas counties in **2010, 2015, and 2020**.  

---

## Problem Statement
Local governments and policymakers need actionable tools to understand how community conditions shape population health.
Our objective was to design an interactive tool that:  
1. Visualizes Health Trends: Tracks changes in Adult Health Score and the relative influence of each SDOH category on health outcomes across Texas counties from 2010, 2015, and 2020.
2. Explores Correlations: Generates correlation matrices between a selected SDOH and health outcomes to identify high-impact factors, as well as explore relationships among SDOH variables.
3. Predict Health Outcomes: Apply Random Forest regression to simulate how county-level changes (like a goal to reduce uninsured rates) could potentially shift its Adult Health Score.

---

## Data Sources  
Our dashboard integrates publicly available datasets from:
- **Agency for Healthcare Research and Quality (AHRQ):** County-level datasets for 2010, 2015, and 2020 containing health and socio-economic indicators (uninsured rate, education attainment, poverty levels, healthcare access, and more). These variables were mapped into the five Social Determinants of Health (SDOH) categories defined by the CDC: Economic Stability, Education, Healthcare Access, Neighborhood & Environment, and Community Context. The variables built into our Adult Health Score outcome variable included injury death rate (physical health) and self-harm death rate (mental health).
- **Federal Reserve Economic Data (FRED):** County-level measures of Premature Death Rates (reflective of overall health) scraped from FRED’s database. These were used to build the Adult Health Score (health outcome) variable in our analysis.

## Repository Data Structure:
- data/
  - SDOH_2010.xlsx, SDOH_2015.xlsx, SDOH_2020.xlsx, TX_County_Premature_Death_Rate.xlsx → Original raw datasets (AHRQ, FRED)
  - SDOH_2010_Final.csv, SDOH_2015_Final.csv, SDOH_2020_Final.csv → Cleaned datasets (ready for analysis)

---

## Approach  
- **Data Cleaning & Integration:** Combined SDOH indicators from AHRQ (2010, 2015, 2020) with premature death rates from FRED, standardized variable names, and exported cleaned datasets.
- **Feature Engineering:** Constructed an outcome variable (Adult Health Score) using weighted mortality rates, normalized for county comparison: 0.6*(Premature Death Rate) + 0.3*(Injury Death Rate) + 0.1*(Self-Harm Death Rate). Normalized for county comparison.
- **SDOH Categorization:** Grouped indicators into CDC’s five SDOH categories. Identified the “Most Influential SDOH” for each county based on z-scores.
- **Building Dashboard:** Built three sections in Plotly Dash.
  - Choropleth Maps: Adult Health Score + Most Influential SDOH by county with interactive hover tooltips, a time slider, and county-level pie charts.
  - Scatterplots & Correlation Heatmaps: Exploring relationships between SDOH variables and health outcomes, as well as correlations between SDOH variables.
  - Predictive Modeling: Trained a Random Forest regression model to predict county-level Adult Health Scores. The five highest-importance SDOH predictors were used as interactive sliders to let users test what-if scenarios and see how policy interventions might impact overall health outcomes.

---

## Geospatial Information

All maps are rendered at the Texas county level. County boundaries are provided through Plotly’s mapping utilities and aligned by FIPS codes in the AHRQ/FRED datasets to ensure consistent spatial joins across datasets and accurate year-to-year comparisons. All interactivity (hover, tooltips, sliders) is allowed using Plotly’s built-in geographic functions.

---

## Key Findings  
- **Economic stability and healthcare access** consistently emerged as the strongest predictors of adult health in Texas.
- Several counties experienced major health gains from 2010 to 2020, while others experienced declines, often correlating with shifts in unemployment and healthcare access.
- The **Random Forest regression model achieved strong predictive accuracy** (RMSE = 0.12), showing that machine learning can identify non-linear SDOH interactions and support evidence-based policymaking to improve community health. 

---

## Repository Contents  
- `code/`  
  - `P08_FinalProjectJupyterNotebook.ipynb` → Jupyter Notebook (EDA, cleaning, modeling)  
  - `texas_dashboard.py` → Dash app script 
- `data/`  
  - `SDOH_2010.xlsx`, `SDOH_2015.xlsx`, `SDOH_2020.xlsx`, `TX_County_Premature_Death_Rate.xlsx` → Original raw datasets  
  - `SDOH_2010_Final.csv`, `SDOH_2015_Final.csv`, `SDOH_2020_Final.csv` → Cleaned, ready-to-use datasets (don't need to run EDA if you download these) 
- `presentation/`
  - `dashboard_demo.mp4` → Walkthrough video of the dashboard
  - `screenshots/` → Key dashboard views (4 images)

---

## Reproducibility

To reproduce or extend our analysis:
- Clone this repository and install required packages:
    - pip install pandas plotly dash scikit-learn
- Use the cleaned datasets in /data (*_Final.csv) for immediate analysis.
- To fully regenerate datasets and models, run:
    - code/P08_FinalProjectJupyterNotebook.ipynb (data cleaning, feature engineering, modeling)
- Launch the dashboard locally with:
    - python code/texas_dashboard.py
- Then open the local URL in your browser to view the dashboard.

## Team Members  
- Quinn Hungerford  
- Shriya Ganesan  
- Amanda Roberts  

---

*Developed as part of CS 329E at The University of Texas at Austin. This project demonstrates how data science and interactive visualization can reveal insights into public health at the county level, supporting data-driven decision making for healthier communities.* 
