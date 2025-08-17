# Load libraries

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import geopandas as gpd
import dash
from dash import dcc, html, Input, Output
import joblib
import json
import requests
import pickle

SDOH_2010 = pd.read_csv('SDOH_2010_Final.csv')
SDOH_2015 = pd.read_csv('SDOH_2015_Final.csv')
SDOH_2020 = pd.read_csv('SDOH_2020_Final.csv')

# Define social determinant categories
sdo_categories = {
    "Economic Stability": ["Median_Household_Income", "Unemployment_Rate", "Income_Inequality_Gini",
                          "Households_on_Food_Stamps", "Households_Income_Below_10K", 
                          "Population_Income_Above_200PCT_Poverty", "Poverty_Rate"],
    "Education": ["High_School_Graduation_Rate", "Bachelor_Degree_Rate", "Less_Than_High_School_Edu", 
                 "Graduate_Degree_Rate", "Youth_Not_in_School_or_Work"],
    "Healthcare Access": ["Uninsured_Rate_Under_64", "Hospitals_with_Ambulance_per_1K", "Median_Distance_to_ER",
                         "Median_Distance_to_Clinic", "Advanced_Nurses_per_1K", "Primary_Care_Shortage_Score",
                         "Households_No_Vehicle"],
    "Neighborhood and Environment": ["Median_Home_Value", "Percent_Homes_Built_Pre_1979", "Days_Heat_Index_Above_100F",
                                    "Storm_Injuries_Total"],
    "Community Context": ["Elderly_Living_Alone", "Single_Parent_Households", "Non_Citizen_Population",
                                  "Limited_English_Households"]
}

# Function to compute category weights with category-level normalization
def compute_category_weights_per_county(df, dataset):
    df = df.copy()

    # Ensure the target variable exists
    if "Adult_Health_Score" not in df.columns:
        raise ValueError("Column 'Adult_Health_Score' not found in dataset. Check column names.")

    # Extract features relevant to categories
    feature_columns = [col for cols in sdo_categories.values() for col in cols if col in df.columns]
    df_features = df[feature_columns].copy()

    # Fill missing values with median
    imputer = SimpleImputer(strategy="median")
    df_features = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns)

    # Normalize features **within each category** (not across all data)
    category_weights = {}
    for category, variables in sdo_categories.items():
        valid_vars = [var for var in variables if var in df_features.columns]
        
        if valid_vars:
            # Extract the dataset's full feature set (to compute mean/std per category)
            dataset_category = dataset[valid_vars].copy()
            
            # Fill missing values in the full dataset before computing scaling
            dataset_category = pd.DataFrame(imputer.fit_transform(dataset_category), columns=dataset_category.columns)

            # Normalize using StandardScaler (Z-score per category)
            scaler = StandardScaler()
            dataset_category_scaled = pd.DataFrame(scaler.fit_transform(dataset_category), columns=dataset_category.columns)

            # Scale the county's values based on the dataset's category scaling
            county_scaled = pd.DataFrame(scaler.transform(df_features[valid_vars]), columns=valid_vars)

            # Sum absolute values of scaled features per category
            category_weights[category] = county_scaled.abs().sum(axis=1).values[0]

    df["Predominant_SDOH"] = df.apply(
        lambda row: max(category_weights, key=category_weights.get), axis=1
    )

    # Normalize category contributions to sum to 100%
    total_weight = sum(category_weights.values())
    if total_weight > 0:
        category_weights = {key: (value / total_weight) * 100 for key, value in category_weights.items()}

    return category_weights if category_weights else None

# Function to plot pie chart for a specific county
def plot_pie_chart_for_county(df, dataset, county_name, year):
    if county_name not in df["County"].values:
        valid_counties = df["County"].unique()
        print(f"County '{county_name}' not found in {year} dataset.")
        print("Available counties:", ", ".join(valid_counties[:10]), "...")  # Show first 10 valid counties
        return
    
    county_data = df[df["County"] == county_name]

    # Compute category contributions for this county using the full dataset for scaling
    category_contributions = compute_category_weights_per_county(county_data, dataset)

    # Plot pie chart
    fig = px.pie(
        names=list(category_contributions.keys()),
        values=list(category_contributions.values()),
        title=f"Social Determinant Contributions to Adult Health Score ({county_name}, {year})",
        hole=0.4
    )

    fig.update_traces(textinfo="percent+label")  # Show percentages and labels on the chart

    fig.show()

max_sdoh_list = []
max_percent_list = []
for county_name in SDOH_2010['County']:
    county_data = SDOH_2010[SDOH_2010["County"] == county_name]

    # Compute category contributions for this county using the full dataset for scaling
    category_contributions = compute_category_weights_per_county(county_data, SDOH_2010)
    max_percent = 0
    max_sdoh = None
    for sdoh, percent in category_contributions.items():
        if percent > max_percent:
            max_percent = percent
            max_sdoh = sdoh
    max_sdoh_list.append(max_sdoh)
    max_percent_list.append(max_percent)
    
SDOH_2010['Max_SDOH_Category'] = max_sdoh_list
SDOH_2010['Max_SDOH_Percent'] = max_percent_list  

max_sdoh_list = []
max_percent_list = []
for county_name in SDOH_2015['County']:
    county_data = SDOH_2015[SDOH_2015["County"] == county_name]

    # Compute category contributions for this county using the full dataset for scaling
    category_contributions = compute_category_weights_per_county(county_data, SDOH_2015)
    max_percent = 0
    max_sdoh = None
    for sdoh, percent in category_contributions.items():
        if percent > max_percent:
            max_percent = percent
            max_sdoh = sdoh
    max_sdoh_list.append(max_sdoh)
    max_percent_list.append(max_percent)
    
SDOH_2015['Max_SDOH_Category'] = max_sdoh_list
SDOH_2015['Max_SDOH_Percent'] = max_percent_list  

max_sdoh_list = []
max_percent_list = []
for county_name in SDOH_2020['County']:
    county_data = SDOH_2020[SDOH_2020["County"] == county_name]

    # Compute category contributions for this county using the full dataset for scaling
    category_contributions = compute_category_weights_per_county(county_data, SDOH_2020)
    max_percent = 0
    max_sdoh = None
    for sdoh, percent in category_contributions.items():
        if percent > max_percent:
            max_percent = percent
            max_sdoh = sdoh
    max_sdoh_list.append(max_sdoh)
    max_percent_list.append(max_percent)
    
SDOH_2020['Max_SDOH_Category'] = max_sdoh_list
SDOH_2020['Max_SDOH_Percent'] = max_percent_list  

# Load data
SDOH_2020 = pd.read_csv("SDOH_2020_Final.csv")
SDOH_2020.tail()

# Define features (SDOH variables) and target (health score)
features = ["Median_Household_Income", "Graduate_Degree_Rate", 
            "Uninsured_Rate_Under_64", "Elderly_Living_Alone", "Median_Home_Value"]

target = "Adult_Health_Score"

X = SDOH_2020[features]
y = SDOH_2020[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Train the model
rf = RandomForestRegressor(n_estimators=68, random_state=8)
rf.fit(X_train, y_train)

joblib.dump(rf, "health_score_model.joblib")

rf_loaded = joblib.load("health_score_model.joblib")

feature_ranges = {
    "Median_Household_Income": (30931, 105956),
    "Graduate_Degree_Rate": (5.95, 20),
    "Uninsured_Rate_Under_64": (0.25, 27.07),
    "Elderly_Living_Alone": (4.90, 20.97),
    "Median_Home_Value": (61800, 378500)
}

# Add year and merge datasets
for df, year in zip([SDOH_2010, SDOH_2015, SDOH_2020], [2010, 2015, 2020]):
    df["Year"] = year

SDOH_all = pd.concat([SDOH_2010, SDOH_2015, SDOH_2020])
SDOH_all["CountyFIPS"] = SDOH_all["CountyFIPS"].astype(str).str.zfill(5)

# Load GeoJSON for Texas counties
texas_geojson = requests.get("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json").json()

# Define SDOH categories
sdo_categories = {
    "Economic Stability": ["Median_Household_Income", "Unemployment_Rate", "Income_Inequality_Gini",
                          "Households_on_Food_Stamps", "Households_Income_Below_10K", 
                          "Population_Income_Above_200PCT_Poverty", "Poverty_Rate"],
    "Education": ["High_School_Graduation_Rate", "Bachelor_Degree_Rate", "Less_Than_High_School_Edu", 
                 "Graduate_Degree_Rate", "Youth_Not_in_School_or_Work"],
    "Healthcare Access": ["Uninsured_Rate_Under_64", "Hospitals_with_Ambulance_per_1K", "Median_Distance_to_ER",
                         "Median_Distance_to_Clinic", "Advanced_Nurses_per_1K", "Primary_Care_Shortage_Score",
                         "Households_No_Vehicle"],
    "Neighborhood and Environment": ["Median_Home_Value", "Percent_Homes_Built_Pre_1979", "Days_Heat_Index_Above_100F",
                                    "Storm_Injuries_Total"],
    "Community Context": ["Elderly_Living_Alone", "Single_Parent_Households", "Non_Citizen_Population",
                                  "Limited_English_Households"]
}

category_colors = {
    "Economic Stability": "#ff928d",
    "Education": "#fbbe7d",
    "Healthcare Access": "#7ed3e3",
    "Neighborhood and Environment": "#a1e2a3",
    "Community Context": "#d6b0f5"
}

# Helper functions
def compute_category_weights_per_county(df, dataset):
    df = df.copy()
    if "Adult_Health_Score" not in df.columns:
        return None

    feature_columns = [col for cols in sdo_categories.values() for col in cols if col in df.columns]
    df_features = df[feature_columns].copy()

    imputer = SimpleImputer(strategy="median")
    df_features = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns)

    category_weights = {}
    for category, variables in sdo_categories.items():
        valid_vars = [var for var in variables if var in df_features.columns]
        if valid_vars:
            dataset_category = dataset[valid_vars].copy()
            dataset_category = pd.DataFrame(imputer.fit_transform(dataset_category), columns=dataset_category.columns)
            scaler = StandardScaler()
            scaler.fit(dataset_category)
            county_scaled = pd.DataFrame(scaler.transform(df_features[valid_vars]), columns=valid_vars)
            category_weights[category] = county_scaled.abs().sum(axis=1).values[0]

    total_weight = sum(category_weights.values())
    if total_weight > 0:
        category_weights = {k: (v / total_weight) * 100 for k, v in category_weights.items()}
    return category_weights

def format_slider_mark(value, feature):
    if "Income" in feature or "Home_Value" in feature:
        return f"${int(value):,}"  # e.g., $31,000
    elif "Elderly_Living_Alone" in feature:
        return f"{round(value, 1)}"
    else:
        return f"{round(value, 1)}%"

# Making dashboard
app = dash.Dash(__name__)
app.title = "Texas Health Dashboard"

app.layout = html.Div([
    html.H1("Texas County Health Dashboard", style={"textAlign": "center"}),
    
    html.Hr(style={"border": "1px solid lightgray", "margin": "20px 0"}),

    html.H3(
        "Texas Counties' Adult Health Scores and Key Social Determinants of Health (SDOH) Over Time",
        style={"textAlign": "center", "marginLeft": "30px"}
    ),
    
    dcc.RadioItems(
        id="map-type",
        options=[
            {"label": "Adult Health Score", "value": "score"},
            {"label": "Most Influential SDOH on Adult Health", "value": "sdoh"}
        ],
        value="score",
        labelStyle={"display": "inline-block", "marginRight": "15px"},
        style={
            "display": "flex",
            "justifyContent": "center",
            "marginBottom": "10px"
        }
    ),

    html.Div(
        dcc.Slider(
            id="year-slider",
            min=2010,
            max=2020,
            step=5,
            marks={2010: "2010", 2015: "2015", 2020: "2020"},
            value=2010
        ),
        style={"paddingLeft": "40px", "paddingRight": "40px"}
    ),

    html.Div([
        html.Div([
            dcc.Graph(id="map", style={"height": "400px"}),
            html.Div(
                id="map-annotation",
                children=[
                    html.Span("Adult Health Score ="),
                    html.Br(),
                    html.Span("0.6*(Premature Death Rate) + 0.3*(Injury Death Rate) + 0.1*(Self-harm Death Rate)")
                ],
                style={
                    "paddingTop": "10px",
                    "fontSize": "16px",
                    "textAlign": "center",
                    "color": "#555"
                }
            )
        ], style={"width": "52%", "display": "inline-block", "padding": "0 10px"}),

        html.Div([
            dcc.Graph(id="pie-chart", style={"height": "400px"}),
            html.Div(
                id="pie-annotation",
                children="SDOH: Social Determinants of Health",
                style={
                    "paddingTop": "10px",
                    "fontSize": "16px",
                    "textAlign": "center",
                    "color": "#555"
                }
            )
        ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"})
    ], style={"display": "flex", "justifyContent": "space-between", "fontSize": "13px"}),

    html.Hr(style={"border": "1px solid lightgray", "margin": "20px 0"}),
    
    html.Div([
        html.Div([
            html.H3(
                "Relationships Between SDOH Variables and Adult Health Outcomes",
                style={"marginLeft": "30px"}
            ),
            html.Label("Select SDOH Category"),
            dcc.Dropdown(
                id="scatter-category",
                options=[{"label": k, "value": k} for k in sdo_categories.keys()],
                value="Economic Stability",
                style={"width": "90%", "marginBottom": "20px"}
            ),
            html.Label("Select SDOH Variable", style={"marginTop": "10px"}),
            dcc.Dropdown(
                id="scatter-variable",
                style={"width": "90%", "marginTop": "5px"}
            ),
            dcc.Graph(id="scatterplot", style={"height": "350px", "width": "300px"})
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "1%", "paddingLeft": "30px"}),
        
        html.Div(style={
            "width": "2%",
            "borderLeft": "1px solid lightgray",
            "height": "650px",
            "margin": "0 5px",
            "display": "inline-block"
        }),
        
        html.Div([
            html.H3(
                "Correlation Matrix For a Selected SDOH",
                style={"marginLeft": "30px"}
            ),
            html.Label("Select SDOH Category"),
            dcc.Dropdown(
                id="category-selector",
                options=[{"label": key, "value": key} for key in sdo_categories.keys()],
                value="Education",
                style={"width": "90%", "marginBottom": "10px"}
            ),
            dcc.Graph(id="correlation-heatmap", style={"height": "350px", "width": "100%"})
        ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "1%"})
    ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"}),

    html.Hr(style={"border": "1px solid lightgray", "margin": "10px 0"}),

    html.Div([
        html.H3(
            "Predict a County's Adult Health Score Based on SDOH Inputs (RMSE = 0.12)",
            style={"textAlign": "center", "marginBottom": "20px"}
        ),

        html.Div([
            html.Div([
                html.Div([
                    html.Label(
                        f"{feature.replace('_', ' ')} "
                        f"{'($)' if 'Income' in feature or 'Home_Value' in feature else '(#)' if feature == 'Elderly_Living_Alone' else '(%)'}"
                        ),
                    dcc.Slider(
                        id=feature,
                        min=feature_ranges[feature][0],
                        max=feature_ranges[feature][1],
                        step=(feature_ranges[feature][1] - feature_ranges[feature][0]) / 100,
                        value=np.mean(feature_ranges[feature]),
                        marks={
                            feature_ranges[feature][0]: format_slider_mark(feature_ranges[feature][0], feature),
                            feature_ranges[feature][1]: format_slider_mark(feature_ranges[feature][1], feature),
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={"marginBottom": "20px"}) for feature in feature_ranges
            ], style={"width": "55%", "paddingRight": "2%"}),

            html.Div([
                dcc.Graph(id="health_score_gauge", style={"height": "350px"})
            ], style={"width": "45%"})
        
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
    ])
])

@app.callback(
    Output("map", "figure"),
    Input("map-type", "value"),
    Input("year-slider", "value")
)
def update_map(map_type, year):
    df_filtered = SDOH_all[SDOH_all["Year"] == year]

    all_fips = [feature["id"] for feature in texas_geojson["features"]]
    all_counties = pd.DataFrame({"CountyFIPS": all_fips})
    df_year = SDOH_all[SDOH_all["Year"] == year]
    df_filtered = all_counties.merge(df_year, on="CountyFIPS", how="left")
    missing_df = df_filtered[df_filtered["Adult_Health_Score"].isna()].copy()
    missing_df = missing_df[missing_df["CountyFIPS"].str.startswith("48")]
    
    if map_type == "score":
        fig1 = px.choropleth(
            df_filtered,
            geojson=texas_geojson,
            locations="CountyFIPS",
            color="Adult_Health_Score",
            hover_name="County",
            color_continuous_scale=["#f64949", "#fcfe87", "#54d961"],
            scope="usa"
        )
        fig1.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                          "🏥 Adult Health Score: <b>%{z:.3f}</b><br>",
            marker_line_color="black",
            marker_line_width=1,
            coloraxis="coloraxis"
        )

        missing_df["dummy"] = 0.001
        fig2 = px.choropleth(
            missing_df,
            geojson=texas_geojson,
            locations="CountyFIPS",
            color="dummy",
            scope="usa"
        )
        fig2.update_traces(
            showscale=False,
            hoverinfo="skip",
            hovertemplate='',  
            marker_line_color='black',
            marker_line_width=1,
            coloraxis=None
        )

        fig = go.Figure(data=fig1.data + fig2.data)

        fig.update_layout(
            title={
                "text": f"Adult Health Score by County in Texas ({year})",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.9,
                "font": {"size": 16}
            },
            coloraxis=dict(
                colorscale=["#f64949", "#fcfe87", "#54d961"],
                colorbar=dict(
                    title="Adult Health Score",
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"],
                    thickness=10,
                    len=0.5,
                    x=0.82
                )
            ),
            margin={"t": 70, "b": 10},
            geo=dict(
                fitbounds="locations",
                visible=False,
                projection_scale=6.7,
                center={"lat": 31.0, "lon": -99.0}
            ),
            width=1000,
            height=700
        )

    else:
        fig1 = px.choropleth(
            df_filtered,
            geojson=texas_geojson,
            locations="CountyFIPS",
            color="Max_SDOH_Category",
            hover_name="County",
            color_discrete_map=category_colors,
            scope="usa",
            labels={"Max_SDOH_Category": ""},
            custom_data=["Max_SDOH_Category"]
        )
        fig1.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                      "🏆 Most Influential SDOH: <b>%{customdata[0]}</b><extra></extra>",
            marker_line_color="black",
            marker_line_width=1
        )

        missing_df["dummy"] = 0.001
        fig2 = px.choropleth(
            missing_df,
            geojson=texas_geojson,
            locations="CountyFIPS",
            color="dummy",
            scope="usa"
        )
        fig2.update_traces(
            showscale=False,
            hoverinfo="skip",
            hovertemplate='',
            marker_line_color='black',
            marker_line_width=1,
            coloraxis=None,
            colorscale=[[0, "#f5f5f5"], [1, "#f5f5f5"]]
        )

        fig = go.Figure(data=fig1.data + fig2.data)

        fig.update_layout(
            title={
                "text": f"Most Influential SDOH by County in Texas ({year})",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.9,
                "font": {"size": 18}
            },
            margin={"t": 70, "b": 10},
            showlegend=False
        )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_scale=7,
        center={"lat": 31.0, "lon": -104.0}
    )

    fig.update_layout(
        geo=dict(
            center={"lat": 31.0, "lon": -106},
            projection_scale=6.7,
        ),
        coloraxis_colorbar=dict(
            title="Adult Health Score",
            thickness=10,
            len=0.5,
            x=0.82,
        ),
        margin={"l": 0, "r": 0, "t": 30, "b": 10},
        width=550,
        height=400
    )

    return fig

@app.callback(
    Output("pie-chart", "figure"),
    Input("map", "clickData"),
    Input("year-slider", "value")
)
def update_pie_chart(clickData, year):
    if clickData is None:
        fig = px.pie()
        fig.update_layout(
            title={
                "text": "Select a county on the map",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16}
            }
        )
        return fig

    county = clickData["points"][0]["hovertext"]
    df_county = SDOH_all[(SDOH_all["County"] == county) & (SDOH_all["Year"] == year)]
    df_dataset = SDOH_all[SDOH_all["Year"] == year]

    if df_county.empty:
        return px.pie(title="Data unavailable")

    weights = compute_category_weights_per_county(df_county, df_dataset)
    if not weights:
        return px.pie(title=f"Invalid data for {county}")

    fig = px.pie(
        names=list(weights.keys()),
        values=list(weights.values()),
        title=f"How Each SDOH Influenced Health in {county} ({year})",
        hole=0.4,
        color=list(weights.keys()),
        color_discrete_map=category_colors,
        category_orders={"names": list(category_colors.keys())}
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>" +
                  "📊 Proportional Influence on Health: <b>%{value:.1f}%</b><extra></extra>"
    )
    fig.update_layout(
        title={
            "text": f"How Each SDOH Influenced Health in {county} ({year})",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16}
        },
        legend=dict(
            orientation="v",
            x=1.0,
            y=-1.0,
            xanchor="left"
        ),
    )
    fig.update_traces(textinfo="percent+label")
    return fig

@app.callback(
    Output("scatter-variable", "options"),
    Output("scatter-variable", "value"),
    Input("scatter-category", "value")
)
def update_variable_dropdown(category):
    variables = sdo_categories.get(category, [])
    options = [{"label": var, "value": var} for var in variables if var in SDOH_all.columns]
    default_value = options[0]["value"] if options else None
    return options, default_value

@app.callback(
    Output("scatterplot", "figure"),
    Input("scatter-variable", "value"),
    Input("year-slider", "value")
)
def update_scatterplot(variable, year):
    if variable is None:
        return px.scatter(title="No variable selected")

    df = SDOH_all[SDOH_all["Year"] == year]
    if variable not in df.columns or "Adult_Health_Score" not in df.columns:
        return px.scatter(title="Data not available")

    selected_category = next(
        (cat for cat, vars in sdo_categories.items() if variable in vars),
        "Economic Stability"
    )

    point_color = category_colors.get(selected_category, "cornflowerblue")

    fig = px.scatter(
        df,
        x=variable,
        y="Adult_Health_Score",
        hover_name="County",
        title=f"Relationship Between {variable}<br>and Adult Health Score ({year})",
        trendline="ols",
        labels={
            variable: variable.replace("_", " "),
            "Adult_Health_Score": "Adult Health Score"
        }
    )

    fig.update_traces(
        marker=dict(
            size=9,
            color=point_color,
            opacity=0.8,
            line=dict(width=0.5, color="white")
        )
    )

    fig.update_layout(
        width=450,
        height=400,
        margin={"t": 90, "b": 40},
        plot_bgcolor="rgba(245, 245, 245, 1)",
        paper_bgcolor="white",
        title={
            "text": f"Relationship Between {variable}<br>and Adult Health Score ({year})",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14}
        },
        xaxis=dict(
            gridcolor="lightgray",
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="lightgray",
            zeroline=False
        )
    )

    return fig

@app.callback(
    Output("correlation-heatmap", "figure"),
    Input("year-slider", "value"),
    Input("category-selector", "value")
)
def update_correlation_matrix(year, selected_category):
    df = SDOH_all[SDOH_all["Year"] == year].copy()

    base_vars = sdo_categories[selected_category]
    other_vars = [v for k, lst in sdo_categories.items() if k != selected_category for v in lst]

    base_vars = [var for var in base_vars if var in df.columns]
    other_vars = [var for var in other_vars if var in df.columns]

    df_clean = df[base_vars + other_vars].dropna()

    if df_clean.empty or not base_vars or not other_vars:
        return px.imshow(np.zeros((1, 1)), labels={"x": "No Data", "y": "No Data"})

    corr_matrix = df_clean[base_vars + other_vars].corr().loc[base_vars, other_vars]

    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale=["#FF0000", "#ffffff", "#006400"],
        zmin=-1,
        zmax=1,
        title=f"Correlation Between {selected_category}<br>and Other SDOH Variables ({year})",
        labels=dict(color="Correlation")
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>vs<br><b>%{x}</b><br>" +
                  "📊 Correlation: <b>%{z:.2f}</b><extra></extra>"
    )
    fig.update_layout(
        width=600,
        height=500,
        margin={"t": 50, "b": 150},
        title_font=dict(size=16),
        xaxis_tickangle=-60,
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        title={
            "text": f"Correlation Between {selected_category}<br>and Other SDOH Variables ({year})",
            "x": 0.5,
            "y": 0.80,
            "xanchor": "center",
            "font": {"size": 14}
        },
        coloraxis_colorbar=dict(
            title="Correlation",
            thickness=10,
            len=0.6,
            x=1.02
        )
    )
    return fig

@app.callback(
    Output("health_score_gauge", "figure"),
    [Input(feature, "value") for feature in feature_ranges]
)
def update_health_score(*values):
    user_input = pd.DataFrame([dict(zip(feature_ranges.keys(), values))])
    health_score = rf_loaded.predict(user_input)[0]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={"text": "Predicted Health Score"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "black",
                   "line": {"color": "black", "width": 2}},
            "steps": [
                {"range": [0, 0.33], "color": "#f64949"},
                {"range": [0.33, 0.66], "color": "#fcfe87"},
                {"range": [0.66, 1], "color": "#54d961"}
            ]
        }
    ))

    return fig
    
if __name__ == "__main__":
    app.run(debug=True, port = '8125')