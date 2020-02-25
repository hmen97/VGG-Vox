# Convert m4a extension files to wav extension files

import os
import argparse
import multiprocessing
from multiprocessing import Process

from pydub import AudioSegment

formats_to_convert = ['.m4a']

def convert():
    for (dirpath, dirnames, filenames) in os.walk("wav/"):
        for filename in filenames:
            if filename.endswith(tuple(formats_to_convert)):

                filepath = dirpath + '/' + filename
                (path, file_extension) = os.path.splitext(filepath)
                file_extension_final = file_extension.replace('.', '')
                try:
                    track = AudioSegment.from_file(filepath,
                                                   file_extension_final)
                    wav_filename = filename.replace(file_extension_final, 'wav')
                    wav_path = dirpath + '/' + wav_filename
                    print('CONVERTING: ' + str(filepath))
                    file_handle = track.export(wav_path, format='wav')
                    os.remove(filepath)
                except:
                    print("ERROR CONVERTING " + str(filepath))

if __name__=="__main__":
    p1 = Process(target=convert)
    p2 = Process(target=convert)
    p3 = Process(target=convert)
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    print("We're done")