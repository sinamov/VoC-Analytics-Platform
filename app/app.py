import streamlit as st
import pandas as pd
import json
import time
from agent import compiled_agent_app  # Imports our LangGraph "brain"
import os
import altair as alt  # For the bar chart
import plotly.express as px # For the new pie chart

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Feedback Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- 2. THE "BEAUTIFUL" CSS ---
st.markdown(r"""
    <style>
    /* Main App Font */
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Inter', sans-serif;
    }
    /* Hide Streamlit Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Card Layout */
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
        margin-bottom: 24px;
    }
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        padding: 12px 16px;
        border-radius: 8px 8px 0 0;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #FF4B4B;
    }
    /* Main Title */
    .st-emotion-cache-1jicfl2 {
        font-size: 32px;
        font-weight: 700;
    }
    /* Custom Button */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    /* Make the app background a light gray */
    .st-emotion-cache-uf99v8 {
        background-color: #F8F9FA;
    }
    /* Hide Uploader Text */
    div[data-testid="stFileUploader"] p, 
    div[data-testid="stFileUploader"] small {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD DATA (for Dashboard) ---
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['aspect'] = df['aspect'].str.lower()
    return df

DATA_PATH = "app/data/dashboard_data.csv"
df_dashboard = load_data(DATA_PATH)

# --- 4. APP LAYOUT ---
st.title("Customer Feedback Analytics")

# --- UPDATED TAB ORDER ---
tab1, tab2, tab3 = st.tabs([
    "Live Sandbox", 
    "Main Dashboard", 
    "Batch Analysis"
])

# --- TAB 1: LIVE SANDBOX (Moved from Tab 2) ---
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Live Agent Analysis")
    st.markdown("Enter a single piece of text (like an email or feedback form) to see the agent analyze it in real-time.")
    
    # Reverted to st.form for simplicity. Since this is now Tab 1,
    # the "page jump" will just reload this same tab.
    form = st.form(key="sandbox_form")
    text_input = form.text_area("Enter email text here:", 
                                value="""Hi team,
The shuffle in the new feature is not reliable and when i want to log in to the structured streaming it crashes. But I have to say that the performance of the dynamic allocation although pretty slow is exceptional. They dynamic allocation must be improved through more memory allocation to the users. They dynamic allocation is very very slow and that hurts performance.

- A Frustrated User""",
                                height=250)
    submit_button = form.form_submit_button(label="Analyze Text")

    if submit_button and text_input:
        with st.spinner("🤖 Agent is thinking..."):
            inputs = {"raw_text": text_input}
            final_state = compiled_agent_app.invoke(inputs)
        
        st.divider()
        st.subheader("Analysis Results")
        
        st.markdown(f"**Summary:** {final_state['summary']}")
        st.markdown(f"**Classification:** {final_state['classification']}")
        
        st.markdown("**Aspect Sentiment:**")
        if final_state.get('absa_results'):
            for item in final_state['absa_results']:
                if item['sentiment'] == 'positive':
                    st.success(f"**{item['aspect'].capitalize()}**: {item['sentiment']}")
                elif item['sentiment'] == 'negative':
                    st.error(f"**{item['aspect'].capitalize()}**: {item.get('sentiment', 'N/A')}")
                else:
                    st.info(f"**{item['aspect'].capitalize()}**: {item.get('sentiment', 'N/A')}")
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: MAIN DASHBOARD (Moved from Tab 1) ---
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if df_dashboard is None:
        st.error("Dashboard data file 'app/data/dashboard_data.csv' not found. Please run the Databricks export notebook.")
    else:
        st.subheader("Product Features with Most No. of Feedbacks")
        
        # --- CHART 1: Top Aspects ---
        aspect_counts = df_dashboard['aspect'].value_counts().nlargest(20).reset_index()
        aspect_counts.columns = ['aspect', 'count']
        
        chart1 = alt.Chart(aspect_counts).mark_bar().encode(
            x=alt.X('aspect', 
                    sort=None, 
                    title="Aspect", 
                    axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('count', title="Total Mentions")
        ).properties(
            title="Top 20 Most Discussed Aspects"
        ).interactive()
        
        st.altair_chart(chart1, use_container_width=True)

        # --- CHART 2: Sentiment-by-Aspect ---
        st.subheader("Sentiment by Aspect")
        
        all_aspects = aspect_counts['aspect'].tolist() 
        selected_aspect = st.selectbox("Choose an aspect to analyze:", all_aspects)
        
        if selected_aspect:
            filtered_df = df_dashboard[df_dashboard['aspect'] == selected_aspect]
            sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Total Mentions", len(filtered_df))
                st.markdown(f"**Positive:** {sentiment_counts[sentiment_counts['sentiment'] == 'positive']['count'].sum()}")
                st.markdown(f"**Neutral:** {sentiment_counts[sentiment_counts['sentiment'] == 'neutral']['count'].sum()}")
                st.markdown(f"**Negative:** {sentiment_counts[sentiment_counts['sentiment'] == 'negative']['count'].sum()}")
                
            with col2:
                color_map = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#17a2b8'}
                fig = px.pie(sentiment_counts, 
                             values='count', 
                             names='sentiment',
                             title=f"Sentiment Breakdown for '{selected_aspect}'",
                             color='sentiment',
                             color_discrete_map=color_map)
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    pull=[0.05 if s == 'negative' else 0 for s in sentiment_counts['sentiment']]
                )
                fig.update_layout(showlegend=False, margin=dict(t=50, b=0, l=0, r=0), font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: BATCH ANALYSIS ---
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Upload a Batch of Emails/Feedbacks")
    
    st.markdown("Upload a CSV file with a `text` column. **Note: File size is programmatically limited to 5MB.**")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 5:
            st.error(f"File size ({file_size_mb:.2f} MB) exceeds 5MB limit.")
        else:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("Error: CSV must contain a column named 'text'.")
            else:
                st.success(f"File '{uploaded_file.name}' uploaded successfully. Found {len(df)} rows.")
                
                if st.button("Start Batch Analysis"):
                    results = []
                    progress_bar = st.progress(0, text="Starting batch analysis...")
                    
                    for i, row in df.iterrows():
                        with st.spinner(f"Analyzing row {i+1}/{len(df)}..."):
                            inputs = {"raw_text": row['text']}
                            final_state = compiled_agent_app.invoke(inputs)
                        
                        results.append({
                            "text": row['text'],
                            "classification": final_state['classification'],
                            "summary": final_state['summary'],
                            "absa_results": json.dumps(final_state['absa_results'])
                        })
                        
                        progress_bar.progress((i + 1) / len(df), text=f"Analyzed row {i+1}/{len(df)}")
                    
                    st.success("Batch analysis complete!")
                    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)