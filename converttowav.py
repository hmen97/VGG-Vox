# Convert m4a extension files to wav extension files

import os
import argparse
import multiprocessing
from multiprocessing import Process
import pandas as pd
from pydub import AudioSegment
formats_to_convert = ['.m4a']


def set_mp(processes=20):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


def convert(filepath):
    (path, file_extension) = os.path.splitext(filepath)
    file_extension_final = file_extension.replace('.', '')
    filename = filepath.split("/")[-1]
    dirpath = filepath.split["/"][:-1]
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
    pool = set_mp(processes=20)
    files = list(pd.read_csv("../meta/data.txt", header=None)[0])
    pool.apply_async(convert, args=(files, ))


