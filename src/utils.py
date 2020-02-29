import librosa
import numpy as np


#install parallel and sox
#converts .wav headers to RIFF
#run this on the from the parent folder ##find . -name '*.WAV' | parallel -P20 sox {} '{.}.wav'##



# def load_wav(vid_path, sr, mode='train'):
#     import tensorflow as tf
#     file = tf.io.read_file(filename=vid_path)
#     tf_wav, tf_sr_ret = tf.audio.decode_wav(contents=file, desired_channels=1)
#     if mode == 'train':
#         tf_extended_wav = tf.compat.v1.concat([tf_wav, tf_wav], 0)
#         if np.random.random() < 0.3:
#             tf_extended_wav = tf.compat.v1.reverse(tf_extended_wav, axis=[0])
#         return tf_extended_wav
#     else:
#         tf_reverse_wav = tf.compat.v1.reverse(tf_wav, axis=[0])
#         tf_extended_wav = tf.compat.v1.concat([tf_wav, tf_reverse_wav], 0)
#         return tf_extended_wav


# def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
#     import tensorflow as tf
#     tf_linear = tf.signal.stft(signals=tf.compat.v1.squeeze(wav), frame_length=win_length,
#                                frame_step=hop_length, fft_length=n_fft, pad_end=False)
#     return tf_linear


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    import tensorflow as tf
    
    #wav = load_wav(path, sr=sr, mode=mode)
    file = tf.io.read_file(filename=path)
    tf_wav, tf_sr_ret = tf.audio.decode_wav(contents=file, desired_channels=1)
    if mode == 'train':
        tf_extended_wav = tf.compat.v1.concat([tf_wav, tf_wav], 0)
        if np.random.random() < 0.3:
            tf_extended_wav = tf.compat.v1.reverse(tf_extended_wav, axis=[0])
    else:
        tf_reverse_wav = tf.compat.v1.reverse(tf_wav, axis=[0])
        tf_extended_wav = tf.compat.v1.concat([tf_wav, tf_reverse_wav], 0)
    ####
    
    #linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    tf_linear = tf.signal.stft(signals=tf.compat.v1.squeeze(tf_extended_wav), frame_length=win_length,
                               frame_step=hop_length, fft_length=n_fft, pad_end=False)
    ####
    
    tf_mag = tf.abs(tf_linear)
    tf_mag_T = tf.compat.v1.transpose(tf_mag)
    tf_freq, tf_time = tf_mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, tf_time-spec_len)
        tf_spec_mag = tf.compat.v1.slice(tf_mag_T, [0, randtime], [tf_freq, spec_len])
    else:
        tf_spec_mag = tf_mag_T
    tf_mu = tf.compat.v1.math.reduce_mean(tf_spec_mag, axis=0, keepdims=True)
    tf_std = tf.compat.v1.math.reduce_std(tf_spec_mag, axis=0, keepdims=True)
    tf_op = (tf_spec_mag-tf_mu)/(tf_std + 1e-5)
    return tf_op.eval(session=tf.compat.v1.Session())


# if __name__=="__main__":
#     x = load_data(path="../../../VGG-Vox/data/dev/aac/id03891/d8eHsoqZGo4/00023.wav", sr=16000)
#     #y = tf.expand_dims(x, -1)
#     y = np.array(3)
#     print("x shape={}, x shape={}".format(type(x), x.shape))

