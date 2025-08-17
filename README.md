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
Our workflow combined data cleaning, exploratory analysis, and predictive modeling:
- **Data Cleaning & Integration:** Combined SDOH indicators from AHRQ (2010, 2015, 2020) with county premature death rates from FRED. Cleaned columns, collapsed poverty rates into one measure, and standardized variable names. Exported final datasets (*_Final.csv).
- **Feature Engineering:** Outcome variable created from mortality rates using Min–Max scaling and weights (lower death rates = higher health). Decided based on research into similar studies to weight it as follows -- 0.6*(Premature Death Rate) + 0.3*(Injury Death Rate) + 0.1*(Self-Harm Death Rate). Normalized for county comparison.
- **SDOH Categorization:** Grouped variables into the five SDOH (Economic, Education, Healthcare, Neighborhood, Community). For each county, calculated the category most atypical (largest z-scores) as the “Most Influential SDOH” dragging the Adult Health Score more different directions.
- **Building Dashboard (Using Plotly Dash):**
  - **First Section:** Built two choropleth maps, both interactive with hover tooltips, time slider (2010–2020), and county-level breakdowns:
     - Adult Health Score by County: Color-coded health outcomes across Texas (from green/good to red/worse), with donut chart showing each SDOH’s contribution for a selected count).
     - Most Influential SDOH by County: Map colored by most influential SDOH category per county, with same donut chart.
  - **Second Section:** Explored SDOH relationships with two different plots, described below.
     - Variable vs. Health Outcome Scatterplot: Select any SDOH variable within a category to see its relationship with Adult Health Score across all Texas counties, displayed as a scatterplot with a regression line. Quick view of which factors are positively/negatively associated with health outcomes.
     - Correlation Matrix Heatmap: Select an SDOH category to view how its variables correlate with one another and with variables from other SDOH categories to identify hidden relationships.
  - **Third Section:** Predictive modeling
    - Trained a Random Forest regression model to predict county-level Adult Health Scores. The five highest-importance SDOH predictors (median household income, uninsured rate, graduate degree rate, elderly living alone, median home value) were used as interactive sliders in the dashboard to let users test what-if scenarios and see how policy interventions might impact overall health outcomes.

---

## Key Findings  
- **Economic stability and healthcare access** consistently emerged as the strongest predictors of adult health in Texas.  
- Some counties showed **substantial health improvements from 2010 → 2020**, while others declined, often correlating with shifts in SDOH variables, specifically unemployment and access to care.  
- Machine learning results indicated that **RandomForest models achieved strong predictive accuracy**, capturing complex, non-linear interactions among SDOH variables with strong predictive accuracy (RMSE = 0.12). This suggests such models can help policymakers identify the most impactful levers and set goals for improving community health.

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

## Team Members  
- Quinn Hungerford  
- Shriya Ganesan  
- Amanda Roberts  

---

*Developed as part of CS 329E at The University of Texas at Austin. This project demonstrates how data science and interactive visualization can reveal insights into public health at the county level, supporting data-driven decision making for healthier communities.* 
