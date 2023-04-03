from pydub import AudioSegment
import os


def convert_file_extension_into_wav(dir_path, filename, overwrite=False):
    formats_to_convert = ['.m4a']

    if filename.endswith(tuple(formats_to_convert)):
        (path, file_extension) = os.path.splitext(filename)
        file_extension = file_extension.replace('.', '')
        wav_filename = filename.replace(file_extension, 'wav')
        wav_file_path = os.path.join(dir_path, wav_filename)

        if not os.path.exists(wav_file_path):
            try:
                track = AudioSegment.from_file(os.path.join(dir_path, filename), format=file_extension)
                print('CONVERTING: ' + str(wav_file_path))
                file_handle = track.export(wav_file_path, format='wav')

                if overwrite:
                    os.remove(os.path.join(dir_path, filename))

            except:
                print("Problem with converting file: {}".format(filename))