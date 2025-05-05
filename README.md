# Jungle Audio Embeddings Project 🌿🐒

This repository contains code and resources for analyzing biodiversity through animal sound embeddings from the Peruvian jungle. This collaborative effort with the San Diego Zoo Research Center uses machine learning techniques to cluster animal sounds, visualize patterns, and build interactive tools.


## 🚀 Features

- **HDBSCAN Clustering**: Density-based clustering to discover groups within jungle audio embeddings.
- **K-means Clustering**: Traditional clustering method for baseline comparisons.
- **Time of Day Analysis**: Investigates sound patterns relative to different times of the day.
- **Spectrogram Generation**: Visualizes animal sounds as spectrograms for intuitive interpretation.
- **Interactive Web App**: Built with Streamlit for real-time exploration of data and clustering results.


## 🗂 Repository Structure

├── data/ # (Not publicly available) Raw and processed audio data – access restricted as the project is ongoing but you can access the streamlit app with the csv and json data .
To access the large file csv you can use: (git lfs install
git clone https://github.com/Sabnam-Pandit/Zoo-Project.git)

├── hdbscan_clustering/ # Scripts and notebooks for HDBSCAN clustering

├── kmeans_clustering/ # Scripts and notebooks for K-means clustering

├── spectrograms/ # Spectrogram generation scripts

├── time_of_day_analysis/ # Analysis scripts based on the time of day

├── streamlit_app/ # Interactive Streamlit web application

└── README.md # Project description and setup instructions

## 🌐 Project Website
You can explore our project through the website below, which includes an embedded Streamlit app and a project overview video:
🔗 [Project Website – Soundscape Analysis](https://sites.google.com/sdsu.edu/soundscape-analysis/home)

## Mentors and Collaborators
Anastasia Kurakova, Sabnam Pandit

Dr. Marie Roch, Department of Computer Science, SDSU

Dr. Hajar Homayouni, Department of Computer Science, SDSU

Dr. Julian Schäfer-Zimmermann, Max Planck Institute

## Acknowledgments
Thanks to the San Diego Zoo Research Center for collaboration and data support.

## License
Distributed under the MIT License. See LICENSE for more information.
