This README provides a structured documentation for your machine learning repository. It covers the project's purpose, dataset details, installation instructions, and a summary of the methodology.

-----

# PRODIGY\_ML\_02: Customer Segmentation using K-Means Clustering

A machine learning project that implements a K-means clustering algorithm to group retail customers based on their purchase history. This segmentation allows businesses to understand customer behavior patterns and devise targeted marketing strategies.

## 📌 Project Overview

Customer segmentation is the process of dividing a company's target market into groups of potential customers with similar needs and behaviors. In this project, we analyze the "Mall Customers" dataset to identify distinct customer segments using the K-Means clustering algorithm.

**Key Goals:**

  * Analyze the distribution of customer demographics (Age, Income, Spending Score).
  * Determine the optimal number of clusters using the Elbow Method.
  * Visualize the clusters to interpret the different customer groups.

## 📂 Dataset

The project uses the **Mall\_Customers.csv** file included in the repository.

  * **Source:** [Mall Customer Segmentation Data](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python) (Commonly used standard dataset).
  * **Features:**
      * `CustomerID`: Unique ID assigned to the customer.
      * `Gender`: Gender of the customer.
      * `Age`: Age of the customer.
      * `Annual Income (k$)`: Annual Income of the customer.
      * `Spending Score (1-100)`: Score assigned by the mall based on customer behavior and spending nature.

## 🛠️ Technologies Used

  * **Python**: Primary programming language.
  * **Pandas**: For data manipulation and analysis.
  * **NumPy**: For numerical operations.
  * **Matplotlib & Seaborn**: For data visualization (plotting clusters, elbow curve, etc.).
  * **Scikit-learn**: For implementing the K-Means clustering algorithm.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Harsh-4210/PRODIGY_ML_02.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd PRODIGY_ML_02
    ```
3.  **Install dependencies:**
    Ensure you have Python installed. You can install the required libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
4.  **Launch the Notebook:**
    Open the Jupyter Notebook to view the code and analysis.
    ```bash
    jupyter notebook "Task 02 – Customer Segmentation Using K-Means Clustering.ipynb"
    ```

## 📊 Methodology

1.  **Data Preprocessing:**

      * Loading the dataset using Pandas.
      * Checking for null values and understanding data structure.
      * Selecting relevant features (typically *Annual Income* and *Spending Score*) for clustering.

2.  **Finding Optimal Clusters (Elbow Method):**

      * Iterating through cluster counts (k=1 to 10).
      * Calculating the Within-Cluster Sum of Squares (WCSS).
      * Plotting the Elbow Graph to identify the "elbow point" where the rate of decrease shifts.

3.  **Model Training:**

      * Initializing the K-Means model with the optimal number of clusters (usually 5 for this dataset).
      * Fitting the model to the data and predicting cluster labels.

4.  **Visualization:**

      * Scatter plots are generated to visualize the customer segments, with centroids marked to represent the center of each cluster.

## 📈 Results

The model typically identifies 5 distinct customer segments (example interpretation):

1.  **Careful**: High Income, Low Spending Score.
2.  **Standard**: Average Income, Average Spending Score.
3.  **Target**: High Income, High Spending Score.
4.  **Careless**: Low Income, High Spending Score.
5.  **Sensible**: Low Income, Low Spending Score.

## 🤝 Contributing

Contributions are welcome\! If you'd like to improve the visualization or try different clustering algorithms (like DBSCAN or Hierarchical Clustering), feel free to fork the repo and submit a pull request.

## 📜 License

This project is open-source and available for educational purposes.
