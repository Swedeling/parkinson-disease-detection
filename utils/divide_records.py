from pydub import AudioSegment
from pydub.silence import split_on_silence
from utils.convert_records import convert_file_extension_into_wav
import os


def divide_records(dir_path):

    for file in os.listdir(dir_path):
        (filename, file_extension) = os.path.splitext(file)

        if file_extension != ".wav":
            convert_file_extension_into_wav(dir_path, file, overwrite=True)

        sound = AudioSegment.from_file(os.path.join(dir_path, filename + ".wav"), format="wav")

        silence_threshold = -50.0  # silence threshold

        sound_chunks = split_on_silence(sound,
                                        min_silence_len=500,  # (ms)
                                        silence_thresh=silence_threshold,
                                        keep_silence=100)  # (ms)

        part1 = sound_chunks[0]
        part2 = sound_chunks[1]
        part3 = sound_chunks[2]

        part1.export(os.path.join(dir_path, filename + "_001.wav"), format="wav")
        part2.export(os.path.join(dir_path, filename + "_002.wav"), format="wav")
        part3.export(os.path.join(dir_path, filename + "_003.wav"), format="wav")

