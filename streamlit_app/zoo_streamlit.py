import streamlit as st
import os
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import namedtuple
import librosa
import soundfile
import numpy as np
from librosa import power_to_db
import json
from collections import Counter
from librosa.util import normalize

st.set_page_config(layout="wide")


# Define the AudioPool_short class (ensure it's included or imported correctly)
class AudioPool_short:
    EntryType = namedtuple("EntryType", ("audio", "adv_N", "len_N", "channel"))

    spec_bias = 1e-7
    default_channel = 0

    def __init__(
        self, adv_s=0.0025, len_s=0.005, window="hamming", low_Hz=0, high_Hz=None
    ):
        self.cache = {}
        self.adv_s = adv_s
        self.len_s = len_s
        self.window = window
        self.low_Hz = low_Hz
        self.high_Hz = high_Hz

    def __getitem__(self, filename):
        if filename not in self.cache:
            sound = soundfile.SoundFile(filename)
            adv_N = int(sound.samplerate * self.adv_s + 0.5)
            len_N = int(sound.samplerate * self.len_s + 0.5)
            channel = 0

            item = self.EntryType(sound, adv_N, len_N, channel)
            self.cache[filename] = item
        return self.cache[filename]

    def get_spectrogram(self, filename, start_s=0, duration_s=-1):
        data = self.get_seconds(filename, start_s, duration_s)
        entry = self.cache[filename]
        Fs = entry.audio.samplerate

        data = normalize(data)

        hop_length = int(self.adv_s * Fs + 0.5)
        win_length = int(self.len_s * Fs + 0.5)

        D = librosa.stft(
            data,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
            center=False,
        )
        spectrogram = np.abs(D)

        spectrogram_db = power_to_db(spectrogram**2, ref=np.max(spectrogram**2))

        return spectrogram_db, Fs

    def get_seconds(self, filename, start_s, duration_s=-1):
        entry = self[filename]
        Fs = entry.audio.samplerate

        if start_s is not None:
            start_sample = int(start_s * Fs)
            current = entry.audio.tell()
            if current != start_sample:
                entry.audio.seek(start_sample)

        Nsamples = int(Fs * duration_s) if duration_s != -1 else -1
        data = entry.audio.read(frames=Nsamples)
        return data


# Function to load the cluster data from a JSON file (you can modify this according to your needs)
def load_cluster_data(json_file_path):
    with open(json_file_path, "r") as f:
        cluster_data = json.load(f)
    return cluster_data


# Function to plot the histograms for the clusters
def plot_cluster_histogram(cluster_to_files, cluster_number, group_by_date=False):
    if str(cluster_number) not in cluster_to_files:
        return None, []  # Return empty list for the x-values

    embeddings = cluster_to_files[str(cluster_number)]

    if group_by_date:
        group_keys = [embedding.split("_")[0] for embedding in embeddings]
    else:
        group_keys = ["_".join(embedding.split("_")[:-1]) for embedding in embeddings]

    key_counts = Counter(group_keys)

    trace = go.Bar(
        x=list(key_counts.keys()),
        y=list(key_counts.values()),
        marker=dict(color="skyblue"),
    )

    return trace, list(key_counts.keys())


# Function to plot the time of day histogram
def plot_time_of_day_histogram(df, cluster_number):
    cluster_df = df[df["cluster"] == cluster_number]

    if cluster_df.empty:
        return None, []  # Return empty list for the x-values

    time_of_day_counts = cluster_df["local_time_of_day"].value_counts()

    trace = go.Bar(
        x=time_of_day_counts.index,
        y=time_of_day_counts.values,
        marker=dict(color="skyblue"),
    )

    return trace, time_of_day_counts.index.tolist()


# Function to plot spectrograms for a given cluster


def plot_cluster_spectrograms_plotly(
    cluster_number, df, audio_folder, frame_duration=0.005
):
    """
    Plot spectrograms for the centroid and three randomly selected embeddings in a cluster,
    ensuring embeddings come from different files when possible.
    """
    if st.button("Refresh"):  # Add a refresh button to reload the spectrograms
        st.rerun()
    # Filter embeddings for the cluster
    cluster_data = df[df["cluster"] == cluster_number]

    if cluster_data.empty:
        raise ValueError(f"No data found for cluster {cluster_number}.")

    # Identify the centroid embedding
    centroid = cluster_data[cluster_data["centroid"] == 1]
    if centroid.empty:
        raise ValueError(f"No centroid found for cluster {cluster_number}.")
    centroid = centroid.iloc[0]
    
    # Extract file name for centroid
   # centroid_file = centroid["embedding"].rsplit("_", 1)[0]

    # Get other embeddings, excluding the centroid
    other_embeddings = cluster_data[cluster_data["centroid"] == 0].copy()

    # Extract file names
    other_embeddings["file_name"] = other_embeddings["embedding"].apply(
        lambda x: x.rsplit("_", 1)[0]
    )


    other_embeddings['embeddings_per_file'] = df.groupby('file_name')['file_name'].transform('count')

    # if len(other_embeddings.groupby("file_name")) < 3:
    #     df_sampled = other_embeddings.sample(n=min(3, len(other_embeddings)))
    # else:
    #     df_sampled = other_embeddings.groupby("file_name").sample(n=1).sample(n=min(3, len(other_embeddings)))

    df_sampled = other_embeddings.sample(n=min(3, len(other_embeddings)), weights='embeddings_per_file')
    selected_embeddings = df_sampled.to_dict(orient="records")

  

    # Combine centroid and selected embeddings
    embeddings_to_plot = [centroid] + selected_embeddings


    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[
            f"{embedding_info['embedding']}" for embedding_info in embeddings_to_plot
        ],
        horizontal_spacing=0.1,
    )

    audio_files_to_plot = set()  # To store unique audio filenames

    for i, embedding_info in enumerate(embeddings_to_plot):
        embedding_name = embedding_info["embedding"]
        audio_filename = embedding_name.rsplit("_", 1)[
            0
        ]  # Remove the index part of the filename
        audio_files_to_plot.add(audio_filename)  # Add only the base filename (unique)

        vector_index = int(embedding_name.rsplit("_", 1)[1])

        audio_file_path = os.path.join(audio_folder, audio_filename)

        start_time = vector_index * frame_duration
        range_start = max(0, start_time - 5)
        range_duration = 10

        audio_duration = 60
        if range_start + range_duration > audio_duration:
            range_duration = audio_duration - range_start
        if range_start < 0:
            range_start = 0
            range_duration = min(10, audio_duration)

        pool = AudioPool_short()
        spectrogram, sr = pool.get_spectrogram(
            audio_file_path, start_s=range_start, duration_s=range_duration
        )

        max_freq_bin = int(20000 / (sr / 2) * spectrogram.shape[0])
        spectrogram_limited = spectrogram[:max_freq_bin, :]

        fig.add_trace(
            go.Heatmap(
                z=spectrogram_limited,
                colorscale="Viridis",
                x=np.linspace(
                    range_start,
                    range_start + range_duration,
                    spectrogram_limited.shape[1],
                ),
                y=np.linspace(0, 20000, max_freq_bin),
                colorbar=dict(title="Amplitude (dB)"),
                showscale=(i == 0),
                name=f"Embedding {embedding_name}",
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=[start_time, start_time],
                y=[0, 20000],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=f"Embedding Center {embedding_name}",
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title=f"Spectrograms for Cluster {cluster_number}",
        title_font=dict(size=16),
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=700,
        width=1200,
    )

    fig.update_annotations(font=dict(size=12))

    st.plotly_chart(fig)

    return embeddings_to_plot


def index_start_end_time(vector_index, frame_duration=0.005):
    start_time = vector_index * frame_duration
    range_start = max(0, start_time - 5)
    range_duration = 10

    audio_duration = 60
    if range_start + range_duration > audio_duration:
        range_duration = audio_duration - range_start
    if range_start < 0:
        range_start = 0
        range_duration = min(10, audio_duration)
    return start_time, range_start, range_duration


def main():
    # Streamlit UI for user input
    st.title("Zoo Data Clustering with Spectrograms")

    col_filter, col_image = st.columns(2, border=False)

    with col_filter:
        # clustering selection
        st.subheader("Please select a clustering method")
        method = st.radio(
            "Please select a clustering method",
            ["HDBSCAN", "K-Means"],
            horizontal=True,
            label_visibility="hidden",
        )

        if method == "HDBSCAN":
            json_file_path = "cluster_to_files_192.json"
            csv_file_path = "embeddings_local.csv"
        else:
            json_file_path = "clustered_embeddings_umap20.json"
            csv_file_path = "updated_embed_df_umap20.csv"

        # Load cluster data and CSV

        cluster_to_files = load_cluster_data(json_file_path)

        st.session_state.number_of_clusters = len(cluster_to_files)

        df = pd.read_csv(csv_file_path)
        audio_folder = "audio"

        cluster_number, group_by_date = cluster_selection()

    with col_image:
        st.image("bird.png")

    if st.session_state.show_histograms:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Cluster Histogram", "Time of Day Histogram"),
        )

        cluster_histogram, cluster_x_values = plot_cluster_histogram(
            cluster_to_files, cluster_number, group_by_date
        )
        if cluster_histogram:
            fig.add_trace(cluster_histogram, row=1, col=1)

        time_of_day_histogram, time_of_day_x_values = plot_time_of_day_histogram(
            df, cluster_number
        )
        if time_of_day_histogram:
            fig.add_trace(time_of_day_histogram, row=1, col=2)

        fig.update_layout(
            title_text=f"Cluster {cluster_number} Histograms",
            showlegend=False,
            height=700,
            width=1200,
            title_x=0.5,
            xaxis=dict(
                type="category",
                tickmode="array",
                tickvals=list(range(len(cluster_x_values))),
                ticktext=cluster_x_values,
            ),
            xaxis2=dict(
                type="category",
                tickmode="array",
                tickvals=list(range(len(time_of_day_x_values))),
                ticktext=time_of_day_x_values,
            ),
        )

        st.plotly_chart(fig)

        spectrogram_plotting(cluster_number, df, audio_folder)


def cluster_selection():
    # Keep track of cluster number in session state to persist across interactions
    if "cluster_number" not in st.session_state:
        st.session_state.cluster_number = 0
    if "show_histograms" not in st.session_state:
        st.session_state.show_histograms = False

    with st.form("initial_form"):
        st.subheader(f"Enter Cluster Number (0-{st.session_state.number_of_clusters}):")

        cluster_number = st.number_input(
            f"Enter Cluster Number (0-{st.session_state.number_of_clusters}):",
            min_value=0,
            step=1,
            value=0,
            label_visibility="hidden",
        )

        group_by_date = st.checkbox("Group by Date")

        show_histogram_button = st.form_submit_button(
            "Show Cluster Histograms and Spectrograms"
        )
        if show_histogram_button:
            st.session_state.cluster_number = (
                cluster_number  # Save the selected cluster in session state
            )

            st.session_state.show_histograms = True
    return cluster_number, group_by_date


def spectrogram_plotting(cluster_number, df, audio_folder):
    embeddings_to_plot = None
    try:
        embeddings_to_plot = plot_cluster_spectrograms_plotly(
            cluster_number, df, audio_folder
        )
    except ValueError as e:
        st.error(f"Error plotting spectrograms: {e}")

    # Audio file selection (does not reset other parts)
    st.markdown("### Play Cluster Audio")
    cols = st.columns(len(embeddings_to_plot) if embeddings_to_plot else 1)
    for i, embedding_info in enumerate(embeddings_to_plot):
        audio_filename = embedding_info["embedding"].rsplit("_", 1)[
            0
        ]  # Remove the index part of the filename
        audio_file_path = os.path.join(audio_folder, audio_filename)

        if os.path.exists(audio_file_path):
            with cols[i]:
                st.write(embedding_info["embedding"])
                start_time, range_start, range_duration = index_start_end_time(
                    int(embedding_info["index"])
                )
                st.audio(
                    audio_file_path,
                    start_time=range_start,
                    end_time=range_start + range_duration,
                )
        else:
            cols[i].warning("Audio file not found.")

    audio_files_to_plot = set()  # To store unique audio filenames
    for embedding_info in df[df["cluster"] == cluster_number].to_dict("records"):
        audio_filename = embedding_info["embedding"].rsplit("_", 1)[
            0
        ]  # Remove the index part of the filename
        audio_files_to_plot.add(audio_filename)


if __name__ == "__main__":
    main()
