from pathlib import Path

from parse_audio import read_audio_files

if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    results = read_audio_files(index, directory)
    print(f"{len(results)} files parsed")
    #convert_to_mel(results[0])
