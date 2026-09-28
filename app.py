import os
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Import trained models and data processing functions from models.py
# Ensure models.py is in the same directory as app.py
from dashboard_models import (
    model_data,
    model as lr_model,
    rf_model,
    xgb_model,
    model_results,
)

# --- 1. DATA PREPARATION FOR STAKEHOLDERS ---
dashboard_df = model_data.copy()

# Ensure Ward and Municipality metadata are cleanly presented
dashboard_df['Ward_Label'] = dashboard_df.apply(
    lambda r: f"Ward {r['WardID']} ({r['Municipality']})", axis=1
)

# Compute live predictions across all models for granular lookup
dashboard_df['Pred_Linear_Regression'] = lr_model.predict(
    dashboard_df[['Youth_Registration_Rate', 'Female_Percentage', 'Female_Male_Ratio']]
).round(2)

dashboard_df['Pred_Random_Forest'] = rf_model.predict(
    dashboard_df[['Youth_Registration_Rate', 'Female_Percentage', 'Female_Male_Ratio']]
).round(2)

dashboard_df['Pred_XGBoost'] = xgb_model.predict(
    dashboard_df[['Youth_Registration_Rate', 'Female_Percentage', 'Female_Male_Ratio']]
).round(2)

dashboard_df['Variance_RF'] = (dashboard_df['Pred_Random_Forest'] - dashboard_df['Turnout_Pct']).round(2)

# --- 2. INITIALIZE DASH APP ---
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "TUT Voter Turnout Predictive Analytics Dashboard"

# --- 3. STYLING (TUT BRAND TRIPLE TONE) ---
TUT_BLUE = "#004B87"
TUT_RED = "#E31B23"
TUT_GOLD = "#FDB913"
BG_COLOR = "#FFFFFF"
CARD_BG = "#F9FAFB"

FONT_FAMILY = "'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif"

card_style = {
    'backgroundColor': CARD_BG,
    'borderRadius': '8px',
    'padding': '20px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.04)',
    'marginBottom': '20px',
    'border': '1px solid #E5E7EB'
}

# --- 4. DASHBOARD LAYOUT ---
app.layout = html.Div(style={'backgroundColor': BG_COLOR, 'fontFamily': FONT_FAMILY, 'minHeight': '100vh', 'color': '#111827'}, children=[
    
    # TOP HEADER WITH LOGO
    html.Div(style={'display': 'flex', 'alignItems': 'center', 'justifySpaceBetween': 'space-between', 
                    'padding': '15px 40px', 'backgroundColor': '#FFFFFF', 'borderBottom': f'4px solid {TUT_BLUE}'}, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '20px'}, children=[
            # Directly displays the official TUT Logo image
            html.Img(src='/assets/SAGEA_affiliate-logos4_TUT.png', style={'height': '65px', 'objectFit': 'contain'}),
            html.Div([
                html.H2("Tshwane University of Technology", style={'margin': '0', 'fontSize': '20px', 'fontWeight': '700', 'color': TUT_BLUE}),
                html.P("Department of Computer Science & Data Analytics | LGE Voter Turnout Intelligence", style={'margin': '2px 0 0 0', 'fontSize': '13px', 'color': '#6B7280'})
            ])
        ]),
        html.Div(style={'textAlign': 'right'}, children=[
            html.Span("Electoral Analytics Dashboard", style={'backgroundColor': '#EFF6FF', 'color': TUT_BLUE, 'padding': '6px 14px', 'borderRadius': '20px', 'fontSize': '12px', 'fontWeight': '600'})
        ])
    ]),

    # MAIN BODY CONTENT
    html.Div(style={'padding': '30px 40px', 'maxWidth': '1400px', 'margin': '0 auto'}, children=[
        
        # TRIPLE TONE EXECUTIVE SUMMARY CARDS
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))', 'gap': '20px', 'marginBottom': '30px'}, children=[
            html.Div(style={**card_style, 'borderLeft': f'6px solid {TUT_BLUE}'}, children=[
                html.P("Total Analyzed Wards", style={'fontSize': '13px', 'color': '#6B7280', 'margin': '0 0 4px 0', 'fontWeight': '600'}),
                html.H3(f"{len(dashboard_df):,}", style={'fontSize': '28px', 'margin': '0', 'fontWeight': '700', 'color': TUT_BLUE}),
                html.P("Across Gauteng & KZN Provinces", style={'fontSize': '12px', 'color': '#9CA3AF', 'margin': '4px 0 0 0'})
            ]),
            html.Div(style={**card_style, 'borderLeft': f'6px solid {TUT_RED}'}, children=[
                html.P("Average 2021 Turnout", style={'fontSize': '13px', 'color': '#6B7280', 'margin': '0 0 4px 0', 'fontWeight': '600'}),
                html.H3(f"{dashboard_df['Turnout_Pct'].mean():.2f}%", style={'fontSize': '28px', 'margin': '0', 'fontWeight': '700', 'color': TUT_RED}),
                html.P(f"Min: {dashboard_df['Turnout_Pct'].min():.1f}% | Max: {dashboard_df['Turnout_Pct'].max():.1f}%", style={'fontSize': '12px', 'color': '#9CA3AF', 'margin': '4px 0 0 0'})
            ]),
            html.Div(style={**card_style, 'borderLeft': f'6px solid {TUT_GOLD}'}, children=[
                html.P("Best Model MAE Error", style={'fontSize': '13px', 'color': '#6B7280', 'margin': '0 0 4px 0', 'fontWeight': '600'}),
                html.H3(f"{model_results['MAE'].min():.2f}%", style={'fontSize': '28px', 'margin': '0', 'fontWeight': '700', 'color': '#D97706'}),
                html.P("Random Forest & XGBoost Performance", style={'fontSize': '12px', 'color': '#9CA3AF', 'margin': '4px 0 0 0'})
            ]),
        ]),

        # INTERACTIVE PREDICTION SIMULATOR & ACTUAL VS PREDICTED
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 2fr', 'gap': '20px', 'marginBottom': '30px'}, children=[
            
            # Interactive Ward Scenario Builder
            html.Div(style=card_style, children=[
                html.H4("Ward Scenario Simulator", style={'marginTop': '0', 'color': TUT_BLUE, 'fontSize': '16px'}),
                html.P("Adjust demographics to forecast expected turnout percentage:", style={'fontSize': '12px', 'color': '#6B7280'}),
                
                html.Label("Youth Registration Rate (18-34) (%)", style={'fontSize': '12px', 'fontWeight': '600'}),
                dcc.Slider(id='input-youth', min=10, max=60, step=0.5, value=30, marks={i: f'{i}%' for i in range(10, 61, 10)}),
                
                html.Div(style={'height': '15px'}),
                html.Label("Female Percentage of Registered Voters (%)", style={'fontSize': '12px', 'fontWeight': '600'}),
                dcc.Slider(id='input-female-pct', min=40, max=70, step=0.5, value=55, marks={i: f'{i}%' for i in range(40, 71, 10)}),
                
                html.Div(style={'height': '15px'}),
                html.Label("Female-to-Male Registered Voter Ratio", style={'fontSize': '12px', 'fontWeight': '600'}),
                dcc.Slider(id='input-fm-ratio', min=0.8, max=1.8, step=0.05, value=1.2, marks={i: f'{i:.1f}' for i in np.arange(0.8, 1.9, 0.2)}),
                
                html.Div(style={'marginTop': '25px', 'padding': '15px', 'backgroundColor': '#EFF6FF', 'borderRadius': '6px', 'textAlign': 'center'}, children=[
                    html.P("Predicted Ward Turnout", style={'fontSize': '12px', 'margin': '0', 'color': TUT_BLUE, 'fontWeight': '600'}),
                    html.H2(id='simulated-prediction-output', style={'margin': '5px 0 0 0', 'color': TUT_BLUE, 'fontSize': '32px'})
                ])
            ]),

            # Actual vs Predicted LGE 2021 Turnout Scatter Chart
            html.Div(style=card_style, children=[
                html.H4("Actual vs. Predicted LGE 2021 Turnout", style={'marginTop': '0', 'color': TUT_BLUE, 'fontSize': '16px'}),
                dcc.Graph(id='actual-vs-predicted-graph', config={'displayModeBar': False}, style={'height': '320px'})
            ])
        ]),

        # GEOGRAPHIC PERFORMANCE & FEATURE INFLUENCE
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '30px'}, children=[
            
            # Lowest vs Highest Municipalities
            html.Div(style=card_style, children=[
                html.H4("2021 Turnout: Lowest vs Highest Municipalities", style={'marginTop': '0', 'color': TUT_BLUE, 'fontSize': '16px'}),
                dcc.Graph(id='muni-comparison-graph', config={'displayModeBar': False}, style={'height': '350px'})
            ]),

            # Feature Importance / Model Weights
            html.Div(style=card_style, children=[
                html.H4("Key Voters Drivers (Predictor Importance)", style={'marginTop': '0', 'color': TUT_BLUE, 'fontSize': '16px'}),
                dcc.Graph(id='feature-importance-graph', config={'displayModeBar': False}, style={'height': '350px'})
            ])
        ]),

        # DETAILED WARD ELECTION RESULTS TABLE FOR STAKEHOLDERS
        html.Div(style=card_style, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}, children=[
                html.H4("Granular Ward Results & Model Predictions Table", style={'margin': '0', 'color': TUT_BLUE, 'fontSize': '16px'}),
                dcc.Dropdown(
                    id='province-filter',
                    options=[{'label': p, 'value': p} for p in sorted(dashboard_df['Province'].unique())],
                    value=None,
                    placeholder="Filter by Province...",
                    style={'width': '220px', 'fontSize': '13px'}
                )
            ]),
            dash_table.DataTable(
                id='ward-results-table',
                columns=[
                    {"name": "Province", "id": "Province"},
                    {"name": "Municipality", "id": "Municipality"},
                    {"name": "Ward ID", "id": "WardID"},
                    {"name": "Youth Reg. Rate (%)", "id": "Youth_Registration_Rate"},
                    {"name": "Female %", "id": "Female_Percentage"},
                    {"name": "F/M Ratio", "id": "Female_Male_Ratio"},
                    {"name": "Actual Turnout (%)", "id": "Turnout_Pct"},
                    {"name": "Pred (Linear)", "id": "Pred_Linear_Regression"},
                    {"name": "Pred (Random Forest)", "id": "Pred_Random_Forest"},
                    {"name": "Pred (XGBoost)", "id": "Pred_XGBoost"},
                    {"name": "RF Error", "id": "Variance_RF"}
                ],
                data=dashboard_df.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': TUT_BLUE,
                    'color': 'white',
                    'fontWeight': '600',
                    'textAlign': 'left',
                    'fontSize': '12px',
                    'padding': '10px'
                },
                style_cell={
                    'textAlign': 'left',
                    'fontSize': '12px',
                    'fontFamily': FONT_FAMILY,
                    'padding': '10px',
                    'border': 'none',
                    'borderBottom': '1px solid #E5E7EB'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#F9FAFB'
                    }
                ]
            )
        ])
    ])
])

# --- 5. CALLBACKS FOR DYNAMIC INTERACTIVITY ---

@app.callback(
    Output('simulated-prediction-output', 'children'),
    [Input('input-youth', 'value'),
     Input('input-female-pct', 'value'),
     Input('input-fm-ratio', 'value')]
)
def update_simulation(youth, female_pct, fm_ratio):
    sample = pd.DataFrame([[youth, female_pct, fm_ratio]], columns=['Youth_Registration_Rate', 'Female_Percentage', 'Female_Male_Ratio'])
    pred = rf_model.predict(sample)[0]
    return f"{pred:.2f}%"

@app.callback(
    Output('actual-vs-predicted-graph', 'figure'),
    Input('province-filter', 'value')
)
def update_actual_vs_predicted(selected_province):
    filtered = dashboard_df if not selected_province else dashboard_df[dashboard_df['Province'] == selected_province]
    
    fig = px.scatter(
        filtered,
        x='Turnout_Pct',
        y='Pred_Random_Forest',
        hover_data=['WardID', 'Municipality'],
        labels={'Turnout_Pct': 'Actual Turnout (%)', 'Pred_Random_Forest': 'Predicted Turnout (%)'},
        color_discrete_sequence=[TUT_BLUE]
    )
    # 1:1 Perfect Prediction Line
    fig.add_trace(go.Scatter(x=[30, 70], y=[30, 70], mode='lines', name='Ideal Fit', line=dict(color=TUT_RED, dash='dash')))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

@app.callback(
    Output('muni-comparison-graph', 'figure'),
    Input('province-filter', 'value')
)
def update_muni_chart(selected_province):
    muni_summary = dashboard_df.groupby(['Municipality'])['Turnout_Pct'].mean().reset_index()
    lowest = muni_summary.sort_values('Turnout_Pct').head(5)
    highest = muni_summary.sort_values('Turnout_Pct', ascending=False).head(5)
    combined = pd.concat([lowest, highest]).sort_values('Turnout_Pct')
    
    colors = [TUT_RED if x in lowest['Municipality'].values else '#10B981' for x in combined['Municipality']]
    
    fig = px.bar(
        combined,
        y='Municipality',
        x='Turnout_Pct',
        orientation='h',
        labels={'Turnout_Pct': 'Average Turnout (%)', 'Municipality': ''}
    )
    fig.update_traces(marker_color=colors)
    fig.add_vline(x=50, line_dash="dash", line_color="gray")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

@app.callback(
    Output('feature-importance-graph', 'figure'),
    Input('province-filter', 'value')
)
def update_feature_importance(_):
    importances = pd.DataFrame({
        'Feature': ['Youth Registration Rate', 'Female-to-Male Ratio', 'Female Percentage'],
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importances,
        y='Feature',
        x='Importance',
        orientation='h',
        labels={'Importance': 'Importance Score', 'Feature': ''}
    )
    fig.update_traces(marker_color=TUT_BLUE)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

@app.callback(
    Output('ward-results-table', 'data'),
    Input('province-filter', 'value')
)
def update_table(selected_province):
    if not selected_province:
        return dashboard_df.to_dict('records')
    return dashboard_df[dashboard_df['Province'] == selected_province].to_dict('records')

# --- 6. SERVER EXECUTION ---
if __name__ == '__main__':
    # Standard deployment on port 8050
    app.run(debug=True, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    app.run_server(debug=False, host='127.0.0.1', port=8050)
    