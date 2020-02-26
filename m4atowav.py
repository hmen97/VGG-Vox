import os
import argparse

from pydub import AudioSegment

formats_to_convert = ['.wav']

def convert():
    for (dirpath, dirnames, filenames) in os.walk("../VGG-Speaker-Recognition/vox1_test_wav/wav/"):
        print(filenames)
        for filename in filenames:
            if filename.endswith(tuple(formats_to_convert)):
                filepath = dirpath + '/' + filename
                (path, file_extension) = os.path.splitext(filepath)
                file_extension_final = file_extension.replace('.', '')
                print("filepath={}, file_extension_final={}".format(filepath, file_extension_final))
                try:
                    track = AudioSegment.from_file(filepath,
                            file_extension_final)
                    wav_filename = filename.replace(file_extension_final, 'm4a')
                    wav_path = dirpath + '/' + wav_filename
                    print('CONVERTING: ' + str(filepath))
                    file_handle = track.export(wav_path, format='m4a')
                    return
                    #os.remove(filepath)
                except:
                    print("ERROR CONVERTING " + str(filepath))
                    return


if __name__=="__main__":
    convert()

