# Convert m4a extension files to wav extension files

import os
from multiprocessing import Pool
import pandas as pd
from pydub import AudioSegment
formats_to_convert = ['.m4a']



def convert(filepath):
    (path, file_extension) = os.path.splitext(filepath)
    file_extension_final = file_extension.replace('.', '')
    filename = filepath.split("/")[-1]
    dirpath = filepath.split("/")[:-1]
    dirpath = "/".join(dirpath)
    #print("filename={}, dirpath={}".format(filename, dirpath))
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
    files = list(pd.read_csv("datam4a.txt", header=None)[0])
    #print(len(files))
    #print(files[0])
    #convert(files[0])
    with Pool(20) as p:
        print(p.map(convert, files))


