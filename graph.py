import matplotlib.pyplot as plt

def graph_spectrogram(file):
    spectrogram_np = file.spectrogram_tensor

    plt.figure(figsize=(8, 5))
    plt.imshow(spectrogram_np.T, aspect="auto", origin="lower", cmap="inferno", extent=[0, 1, 0, 1])
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    