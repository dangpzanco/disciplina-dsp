import numpy as np
import librosa




def PitchProfileFreq(spec, f, fd):
    spec = np.abs(spec)
    upcp = np.zeros(136, spec.shape[-1])
    for i in range(24,136):
        p_index = (f > fd[i]) & (f < fd[i+1])
        upcp[i,] = spec[p_index,].max(axis=0)

    return upcp



def PitchProfileQuef(acr, f, fd):
    acr = np.abs(acr)
    upcp = np.zeros(136, acr.shape[-1])
    for i in range(112):
        p_index = (f > fd[i]) & (f < fd[i+1])
        upcp[i,] = acr[p_index,].max(axis=0)

    return upcp


def IsCandidate(s, c, pitch, num_s, num_c, ratio, NumPerOctave):
    isornot = 0
    har = np.array([0, 12, 19, 24, 28, 31, 34, 36, 38, 40]) * NumPerOctave/12

    cond1 = s[pitch+har[:num_s]].min() > 0 and c[pitch-har[:num_c]].min() > 0
    cond2 = np.count_nonzero(s[pitch:pitch+har[num_s-1]]) >= ratio * (har[num_s-1]+1)
    cond3 = np.count_nonzero(c[pitch:-1:pitch-har[num_c-1]]) >= ratio * (har[num_c-1]+1)

    # if (min(s(pitch+har(1:num_s)))>0 && min(c(pitch-har(1:num_c)))>0)
    #         && ~(nnz(s(pitch:pitch+har(num_s)))>=ratio*(har(num_s)+1) && nnz(c(pitch:-1:pitch-har(num_c))) >= ratio * (har(num_c)+1))
    #     isornot=1

    if cond1 and not (cond2 and cond3):
        isornot = 1

    return isornot



# function [tfr, ceps, GCoS, upcp, upcpt, upcp_final, t] = CFP_GCoS(x, fr, fs, Hop, h, g1, g2)
def CFP_GCoS(x, fr, fs, Hop, h, g1, g2):
    fc = 20
    tc = 1/20000

    [tfr, f, t, N] = STFT(x, fr, fs, Hop, h);
    tfr = abs(tfr).^g1;

    fc_idx = round(fc/fr);
    tc_idx = round(fs*tc);

    tfr = nonlinear_func(abs(tfr), g1, fc_idx);
    ceps = real(fft(tfr))./sqrt(N);
    ceps = nonlinear_func(ceps, g2, tc_idx);

    GCoS = real(fft(ceps))./sqrt(N);
    GCoS = nonlinear_func(GCoS, 1, fc_idx);

    tfr = tfr(1:round(N/2),:);
    ceps = ceps(1:round(N/2),:);
    GCoS = GCoS(1:round(N/2),:);

    HighFreqIdx = round((1/tc)/fr)+1;
    f = f(1:HighFreqIdx);
    tfr = tfr(1:HighFreqIdx,:);
    HighQuefIdx = round(fs/fc)+1;
    q = (0:HighQuefIdx-1)./fs;
    ceps = ceps(1:HighQuefIdx,:);

    GCoS = PeakPicking(GCoS);
    ceps = PeakPicking(ceps);

    % tceps = CepsConvertFreq(ceps, f, fs);

    midi_num=-3:133;
    fd=440*2.^((midi_num-69-0.5)/12);
    upcp = PitchProfileFreq(GCoS, f, fd);
    upcpt = PitchProfileQuef(ceps, 1./q, fd);

    [upcp, upcpt, upcp_final] = PitchFusion(upcp, upcpt, 4, 4, 0.7, 12);

    return results





print('hi')


