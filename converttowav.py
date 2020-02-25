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
    p4 = Process(target=convert)
    p5 = Process(target=convert)
    p6 = Process(target=convert)
    p7 = Process(target=convert)
    p8 = Process(target=convert)
    p9 = Process(target=convert)
    p10 = Process(target=convert)
    p11 = Process(target=convert)
    p12 = Process(target=convert)
    p13 = Process(target=convert)
    p14 = Process(target=convert)
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    print("We're done")
