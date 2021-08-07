import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl

event_name = 'GW150914'

make_plots = 1

events = json.load(open("BBH_events_v3.json", "r"))
event = events[event_name]
fn_H1 = event['fn_H1']  # H1（ハンフォード）で観測されたデータ
fn_L1 = event['fn_L1']  # L1（リビングストン）で観測されたデータ
fn_template = event['fn_template']  # 予想されるテンプレートのデータ
fs = event['fs']  # サンプルレート（今回は4096）
tevent = event['tevent']  # 重力波のイベントの起こった時間
fband = event['fband']  # バンドパスシグナルの周波数帯
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')

time = time_H1
dt = time[1] - time[0]

deltat = 5

NFFT = 4 * fs
Pxx_H1, freqs = mlab.psd(strain_H1, Fs=fs, NFFT=NFFT)
psd_H1 = interp1d(freqs, Pxx_H1)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs=fs, NFFT=NFFT)
psd_L1 = interp1d(freqs, Pxx_L1)


def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)

    hf = np.fft.rfft(strain)
    norm = 1. / np.sqrt(1. / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


strain_H1_whiten = whiten(strain_H1, psd_H1, dt)
strain_L1_whiten = whiten(strain_L1, psd_L1, dt)


bb, ab = butter(4, [fband[0] * 2. / fs, fband[1] * 2. / fs], btype='band')
normalization = np.sqrt((fband[1] - fband[0]) / (fs / 2))
strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
strain_L1_whitenbp = filtfilt(bb, ab, strain_L1_whiten) / normalization

psd_window = np.blackman(NFFT)
NOVL = NFFT / 2

f_template = h5py.File(fn_template, "r")
template_p, template_c = f_template["template"][...]

f_template.close()
template_offset = 16

template = (template_p + template_c * 1.j)
etime = time + template_offset
datafreq = np.fft.fftfreq(template.size) * fs
df = np.abs(datafreq[1] - datafreq[0])

dwindow = signal.tukey(template.size, alpha=1. / 8)

template_fft = np.fft.fft(template * dwindow) / fs

det = 'H1'
data = strain_H1.copy()
data_psd, freqs = mlab.psd(data, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)

data_fft = np.fft.fft(data * dwindow) / fs

power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
optimal = data_fft * template_fft.conjugate() / power_vec
optimal_time = 2 * np.fft.ifft(optimal) * fs
sigmasq = 1 * (template_fft * template_fft.conjugate() / power_vec).sum() * df
sigma = np.sqrt(np.abs(sigmasq))
SNR_complex = optimal_time / sigma

peaksample = int(data.size / 2)
SNR_complex = np.roll(SNR_complex, peaksample)
SNR = abs(SNR_complex)

indmax = np.argmax(SNR)
timemax = time[indmax]
SNRmax = SNR[indmax]

d_eff = sigma / SNRmax
horizon = sigma / 8

phase = np.angle(SNR_complex[indmax])
offset = (indmax - peaksample)

template_phaseshifted = np.real(template * np.exp(1j * phase))
template_rolled = np.roll(template_phaseshifted, offset) / d_eff

template_whitened = whiten(template_rolled, interp1d(freqs, data_psd), dt)
template_match = filtfilt(bb, ab, template_whitened) / normalization

strain_whitenbp = strain_L1_whitenbp
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(time - tevent, strain_whitenbp, "b", label='L1 whitened h(t)')
plt.plot(time - tevent - 0.007, -strain_H1_whitenbp, "r", label=' H1whitened h(t)')
plt.plot(time - tevent - 0.004, template_match, 'k', label='Template(t)')
plt.ylim([-10, 10])
plt.xlim([-0.15, 0.05])
plt.grid('on')
plt.xlabel('Time (s)')
plt.ylabel('whitened strain (units of noise stdev)')
plt.legend(loc='upper left')
plt.show()
