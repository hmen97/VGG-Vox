import librosa
import numpy as np
import tensorflow as tf

#install parallel and sox
#converts .wav headers to RIFF
#run this on the from the parent folder ##find . -name '*.WAV' | parallel -P20 sox {} '{.}.wav'##



def load_wav(vid_path, sr, mode='train'):
    #wav, sr_ret = librosa.load(vid_path, sr=sr)
    #print("wav_shape={}, sr_ret={}".format(wav.shape, sr_ret))
    file = tf.io.read_file(filename=vid_path)
    tf_wav, tf_sr_ret = tf.audio.decode_wav(contents=file, desired_channels=1)
    #tf_wav_np = tf_wav.eval(session=tf.compat.v1.Session())
    #print("tf_wav={}, tf_sr_ret={}".format(tf_wav_np.shape, tf_sr_ret))
    tf_sr_ret_np = tf_sr_ret.eval(session=tf.compat.v1.Session())
    assert tf_sr_ret_np == sr
    if mode == 'train':
        #appending the wav file to itself
        #extended_wav = np.append(wav, wav)
        tf_extended_wav = tf.compat.v1.concat([tf_wav, tf_wav], 0)
        #tf_extended_wav_np = tf_extended_wav.eval(session=tf.compat.v1.Session())
        #print("extended_wav{}".format(extended_wav.shape))
        #print("tf_extended_wav{}".format(tf_extended_wav_np.shape))
        if np.random.random() < 0.3:
            #reversing the audio randomly
            #extended_wav = extended_wav[::-1]
            #print("extended_wav_reverse{}".format(extended_wav.shape))
            tf_extended_wav = tf.compat.v1.reverse(tf_extended_wav, axis=[0])
            #tf_reverse_wav_np = tf_reverse_wav.eval(session=tf.compat.v1.Session())
            # print("tf_reverse_wav_np{}".format(tf_reverse_wav_np.shape))
            # print("exwav-tf_exwav={}".format(np.array_equal(np.expand_dims(extended_wav, -1), tf_reverse_wav_np)))
            #print("tf_extended_wav={}".format(tf_extended_wav.shape))
        return tf_extended_wav
    else:
        #appending the reversed audio input to original
        #extended_wav = np.append(wav, wav[::-1])
        tf_reverse_wav = tf.compat.v1.reverse(tf_wav, axis=[0])
        tf_extended_wav = tf.compat.v1.concat([tf_wav, tf_reverse_wav], 0)
        return tf_extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    # linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    # tf_wav = tf.convert_to_tensor(wav, np.float32)
    #print("tf_wav={}".format(tf_wav))
    tf_linear = tf.signal.stft(signals=tf.compat.v1.squeeze(wav), frame_length=win_length,
                               frame_step=hop_length, fft_length=n_fft, pad_end=False)
    # print("tf_linear={}".format(tf.transpose(tf_linear).shape))
    # print("linear={}".format(linear.shape))
    # tflinear = tf_linear.eval(session=tf.compat.v1.Session())
    # print("tf_linear={}".format(type(tf_linear)))
    #print("tflinear={}, linear".format(tf_linear.shape))
    return tf_linear


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    #print("wav={}".format(wav.shape))
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    #mag, _ = librosa.magphase(linear_spect)  # magnitude
    tf_mag = tf.abs(linear_spect)
    #tf_mag_np = tf_mag.eval(session=tf.compat.v1.Session())
    #print("mag={} tf_mag={} equal={}".format(mag.shape, tf_mag_np.shape, np.array_equal(mag, tf_mag_np)))
    #mag_T = mag.T
    tf_mag_T = tf.compat.v1.transpose(tf_mag)
    #tf_mag_T_np = tf_mag_T.eval(session=tf.compat.v1.Session())
    #print("mag_T={} tfmagT={} equal={}".format(mag_T.shape, tf_mag_T_np.shape, np.array_equal(mag_T, tf_mag_T_np)))
    #freq, time = mag_T.shape
    #print("tf_mag_T={}".format(tf_mag_T.shape))
    tf_freq, tf_time = tf_mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, tf_time-spec_len)
        #spec_mag = mag_T[:, randtime:randtime+spec_len]
        #print("random_time={}".format(randtime))
        tf_spec_mag = tf.compat.v1.slice(tf_mag_T, [0, randtime], [tf_freq, spec_len])
        #tf_spec_mag_np = tf_spec_mag.eval(session=tf.compat.v1.Session())
        #print("spec_mag={} tf_spec_mag={} equal={}".format(spec_mag.shape, tf_spec_mag.shape, np.array_equal(spec_mag, tf_spec_mag_np)))
    else:
        #spec_mag = mag_T
        tf_spec_mag = tf_mag_T
    # preprocessing, subtract mean, divided by time-wise var
    #mu = np.mean(spec_mag, 0, keepdims=True, dtype=np.float32)
    #print("mu_avg={}".format(np.average(mu)))
    #std = np.std(spec_mag, 0, keepdims=True, dtype=np.float32)
    #op = (spec_mag - mu) / (std + 1e-5)
    tf_mu = tf.compat.v1.math.reduce_mean(tf_spec_mag, axis=0, keepdims=True)
    #tf_mu_np = tf_mu.eval(session=tf.compat.v1.Session())
    #print("mu={}, tf_mu_np={}, equal={}".format(mu.shape, tf_mu_np.shape, np.average(np.subtract(mu, tf_mu_np))))
    tf_std = tf.compat.v1.math.reduce_std(tf_spec_mag, axis=0, keepdims=True)
    # tf_std_np = tf_std.eval(session=tf.compat.v1.Session())
    # print("std={}, tf_std={}, equal={}".format(std.shape, tf_std_np.shape, np.average(np.subtract(std, tf_std_np))))
    tf_op = (tf_spec_mag-tf_mu)/(tf_std + 1e-5)
    tf_op_np = tf_op.eval(session=tf.compat.v1.Session())
    # print("op={}, tf_op_np={}, equal={}".format(op.shape, tf_op_np.shape, np.average(np.subtract(op, tf_op_np))))
    return tf_op_np


if __name__=="__main__":
    x = load_data(path="../wav/00313.wav", sr=16000)
    y = np.expand_dims(np.expand_dims(x, 0), -1)
    print("x shape={}, y shape={}".format(x.shape, y.shape))

