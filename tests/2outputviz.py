import pandas as pd
import os
import re

def process_and_save_file(file_path):
    df = pd.read_excel(file_path)
    
    # Remove 'Column1.' prefix from column names
    df.columns = [col.replace('Column1.', '') for col in df.columns]
    
    # Standardize the 'hit' column based on 'hitRelevance'
    df['hit'] = df['hitRelevance'].apply(lambda x: True if x >= 0.5 else False)
    
    # Extract metadata from filename
    filename = os.path.basename(file_path)
    match = re.search(r'run(\d+)_(gpt3-5|gpt4o)_(?i)(businessAnalyst|developer|static|tester)', filename)
    if match:
        df['run'] = match.group(1)
        df['model'] = match.group(2)
        df['persona'] = match.group(3).lower()
    else:
        df['run'] = 'unknown'
        df['model'] = 'unknown'
        df['persona'] = 'unknown'
        print(f"Warning: Couldn't parse metadata for file {file_path}")
    
    # Save the modified DataFrame back to the Excel file
    df.to_excel(file_path, index=False)
    
    return df

def process_all_files(file_paths):
    all_data = []
    for file_path in file_paths:
        df = process_and_save_file(file_path)
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

# List of file paths
file_paths = [
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run1_gpt3-5_businessAnalyst.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run1_gpt3-5_developer.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run1_gpt3-5_static.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run1_gpt3-5_tester.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run1_gpt4o_businessAnalyst.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run1_gpt4o_developer.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run1_gpt4o_static.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run1_gpt4o_tester.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run2_gpt3-5_businessAnalyst.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run2_gpt3-5_developer.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run2_gpt3-5_static.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run2_gpt3-5_tester.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run2_gpt4o_businessAnalyst.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run2_gpt4o_developer.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run2_gpt4o_Static.xlsx",
    r"D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\test output\run2_gpt4o_tester.xlsx"
]

# Process all files and get combined data
combined_df = process_all_files(file_paths)

print("All files have been processed and updated.")

# Print summary of processed data
print("\nSummary of processed data:")
print(combined_df.groupby(['run', 'model', 'persona', 'hit']).size().reset_index(name='count'))



import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Function to load all Excel files
def load_files(file_paths):
    dataframes = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        # Remove 'Column1.' prefix from column names
        df.columns = [col.replace('Column1.', '') for col in df.columns]
        # Add filename as a column
        df['filename'] = os.path.basename(file_path)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


df = load_files(file_paths)

# Set a consistent color palette and theme
color_palette = px.colors.qualitative.Pastel
theme_layout = dict(
    font=dict(family="Arial", size=12),
    plot_bgcolor='rgba(240,240,240,0.8)',
    paper_bgcolor='rgba(240,240,240,0.8)',
    title_font=dict(size=20, color="#333333"),
    legend_title_font=dict(size=14),
    legend_font=dict(size=12),
)

# 1. Number of hits (True/False)
def plot_hit_counts():
    hit_counts = df.groupby('filename')['hit'].value_counts().unstack()
    fig = px.bar(hit_counts, barmode='group', color_discrete_sequence=color_palette)
    fig.update_layout(
        title="Number of Hits by File",
        xaxis_title="Filename",
        yaxis_title="Count",
        legend_title="Hit",
        **theme_layout
    )
    fig.show()

# 2. HitRelevance box plots
def plot_hit_relevance_boxplots():
    fig = px.box(df, y='hitRelevance', x='filename', color='filename',
                 color_discrete_sequence=color_palette)
    fig.update_layout(title="Hit Relevance Distribution by File", **theme_layout)
    fig.show()

# 3. Gemini Score
def plot_gemini_score():
    fig = px.histogram(df, x='gemini_evaluation', color='filename',
                       color_discrete_sequence=color_palette)
    fig.update_layout(title="Gemini Score Distribution", **theme_layout)
    fig.show()

# 4. Follow up on topic
def plot_follow_up_on_topic():
    df['follow_up_on_topic'] = df['follow_up_on_topic'].str.contains('yes', case=False)
    follow_up_counts = df.groupby('filename')['follow_up_on_topic'].value_counts().unstack()
    fig = px.bar(follow_up_counts, barmode='group', color_discrete_sequence=color_palette)
    fig.update_layout(
        title="Follow-up on Topic by File",
        xaxis_title="Filename",
        yaxis_title="Count",
        legend_title="Follow-up on Topic",
        **theme_layout
    )
    fig.show()

# 5. HitRelevance vs Gemini score
def plot_hit_relevance_vs_gemini():
    fig = px.scatter(df, x='hitRelevance', y='gemini_evaluation', color='filename',
                     color_discrete_sequence=color_palette)
    fig.update_layout(title="Hit Relevance vs Gemini Score", **theme_layout)
    fig.show()

# 6. Average HitRelevance
def plot_avg_hit_relevance():
    avg_hit_relevance = df.groupby('filename')['hitRelevance'].mean().reset_index()
    fig = px.bar(avg_hit_relevance, x='filename', y='hitRelevance',
                 color_discrete_sequence=color_palette)
    fig.update_layout(title="Average Hit Relevance by File", **theme_layout)
    fig.show()

# 7. Unique questions generated in each file
def plot_unique_questions():
    unique_questions = df.groupby('filename')['question'].nunique().reset_index()
    fig = px.bar(unique_questions, x='filename', y='question',
                 color_discrete_sequence=color_palette)
    fig.update_layout(
        title="Unique Questions Generated by File",
        xaxis_title="Filename",
        yaxis_title="Number of Unique Questions",
        **theme_layout
    )
    fig.show()

# 8. Unique summaries counts compared
def plot_unique_summaries():
    unique_summaries = df.groupby('filename')['summary'].nunique().reset_index()
    fig = px.bar(unique_summaries, x='filename', y='summary',
                 color_discrete_sequence=color_palette)
    fig.update_layout(
        title="Unique Summaries Count by File",
        xaxis_title="Filename",
        yaxis_title="Number of Unique Summaries",
        **theme_layout
    )
    fig.show()

# Generate all visualizations
plot_hit_counts()
plot_hit_relevance_boxplots()
plot_gemini_score()
plot_follow_up_on_topic()
plot_hit_relevance_vs_gemini()
plot_avg_hit_relevance()
plot_unique_questions()
plot_unique_summaries()