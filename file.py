import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder  
import base64 

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #ffffff; background-color: #4CAF50; padding: 10px; text-decoration: none; border-radius: 5px; margin: 10px; display: inline-block;">{text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def bar_plot(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    value_counts = df[column].value_counts()
    y_range = range(0, max(value_counts) + 1)
    ax.bar(value_counts.index, value_counts.values)

    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Bar Plot for {column}')
    plt.yticks(y_range)

    st.pyplot(fig)

def scatter_plot(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[column], range(len(df)))

    plt.xlabel(column)
    plt.ylabel('Index')
    plt.title(f'Scatter Plot for {column}')
    
    st.pyplot(fig)


def main():
    # Set Streamlit app title and page config
    st.set_page_config(
        page_title="Data Manipulation Application",
        page_icon=":pencil:",
        layout="wide",
    )

    # Dark theme with custom background color
    st.markdown(
        """
        <style>
        body {
            color: #ffffff;
            background-color: #1a1a1a;
        }
        .stButton>button {
            color: #ffffff;
            background-color: #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Broader heading
    st.title("Data Manipulation Application")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Display basic information about the dataset
        st.write("### Dataset Information:")
        st.write(df.info())

        # Display Column Names
        st.write("### Column Names:")
        column_names = df.columns.tolist()
        st.write(column_names)

        # Visualization Section
        st.write("## Data Visualization")

        # Select Visualization Type
        visualization_type = st.selectbox("Select Visualization Type", ["Bar Plot", "Scatter Plot"])

        # Select Columns for Visualization
        selected_column = st.selectbox("Select Column", column_names)

        # Generate Plot based on user inputs
        if st.button("Generate Plot"):
          if visualization_type == "Bar Plot":
            bar_plot(df, selected_column)
          elif visualization_type == "Scatter Plot":
            scatter_plot(df, selected_column)
        # Data Cleaning and Preprocessing
        st.write("## Data Cleaning and Preprocessing")

        # Handling Missing Values
        if st.checkbox("Handle Missing Values"):
            st.write("### Handling Missing Values:")
            st.write("Original Dataset:")
            st.write(df)

            # Fill missing values with mean for numeric columns only
            numeric_columns = df.select_dtypes(include=['number']).columns
            df_filled = df.copy()
            for col in numeric_columns:
                df_filled[col].fillna(df[col].mean(), inplace=True)

            st.write("Dataset after handling missing values (filled with mean):")
            st.write(df_filled)

            # Download button for the filled dataset
            if st.button("Download Filled Dataset", key="download_filled_dataset_btn"):
                download_link(df_filled, 'filled_dataset.csv', 'Download Filled Dataset')

        # Removing Outliers
        if st.checkbox("Remove Outliers"):
            st.write("### Removing Outliers:")
            st.write("Original Dataset:")
            st.write(df)

            # Remove outliers using z-score for numeric columns only
            numeric_columns = df.select_dtypes(include=['number']).columns
            df_no_outliers = df.copy()
            for col in numeric_columns:
                z_scores = (df[col] - df[col].mean()) / df[col].std()
                df_no_outliers = df_no_outliers[(z_scores < 3) & (z_scores > -3)]

            st.write("Dataset after removing outliers (z-score method):")
            st.write(df_no_outliers)

            # Download button for the dataset without outliers
            if st.button("Download Dataset (No Outliers)", key="download_no_outliers_dataset_btn"):
                download_link(df_no_outliers, 'no_outliers_dataset.csv', 'Download Dataset (No Outliers)')

        # Encode Categorical Data
        if st.checkbox("Encode Categorical Data"):
            st.write("### Encoding Categorical Data:")
            st.write("Original Dataset:")
            st.write(df)

            # Check if there are categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            if not categorical_columns.empty:
                # Ask user for encoding method
                encoding_method = st.radio("Select encoding method:", ["Label Encoding"])

                # Perform encoding based on user selection
                if encoding_method == "Label Encoding":
                    df_encoded = df.copy()
                    label_encoder = LabelEncoder()
                    for col in categorical_columns:
                        df_encoded[col] = label_encoder.fit_transform(df[col])

                    st.write("Dataset after label encoding:")
                    st.write(df_encoded)

                    # Download button for the label-encoded dataset
                    if st.button("Download Label Encoded Dataset", key="download_label_encoded_dataset_btn"):
                        download_link(df_encoded, 'label_encoded_dataset.csv', 'Download Label Encoded Dataset')
            else:
                st.warning("No categorical columns found.")

if __name__ == "__main__":
    main()