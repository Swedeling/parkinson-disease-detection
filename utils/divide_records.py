from pydub import AudioSegment
from pydub.silence import split_on_silence
from convert_records import convert_file_extension_into_wav
import os


def divide_records(dir_path):

    for file in os.listdir(dir_path):
        if file != "divided":
            (filename, file_extension) = os.path.splitext(file)

            if file_extension != ".wav":
                convert_file_extension_into_wav(dir_path, file, overwrite=True)

            sound = AudioSegment.from_file(os.path.join(dir_path, filename + ".wav"), format="wav")

            silence_threshold = -40  # silence threshold

            sound_chunks = split_on_silence(sound,
                                            min_silence_len=200,  # (ms)
                                            silence_thresh=silence_threshold,
                                            keep_silence=100)  # (ms)

            for idx, part in enumerate(sound_chunks):
                part.export(os.path.join(dir_path, "divided", filename + "_00{}.wav".format(idx + 1)), format="wav")

