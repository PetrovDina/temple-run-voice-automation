import soundfile as sf
from pedalboard import (
    Pedalboard,
    Chorus,
    Distortion,
    LadderFilter,
    LowpassFilter,
    Phaser,
    PitchShift,
    Reverb,
)

from pathlib import Path


def chorus(audio, sample_rate, directory, id):
    board = Pedalboard([
        Chorus(depth=0.5),
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-chorus.wav'.format(directory, id), 'w', samplerate=sample_rate, channels=len(effected.shape)) as f:
        f.write(effected)


def distortion(audio, sample_rate, directory, id):
    board = Pedalboard([
        Distortion(drive_db=30),
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-distortion.wav'.format(directory, id), 'w', samplerate=sample_rate,
                      channels=len(effected.shape)) as f:
        f.write(effected)


def highpass_filter(audio, sample_rate, directory, id):
    board = Pedalboard([
        LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=1000),
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-hpf.wav'.format(directory, id), 'w', samplerate=sample_rate,
                      channels=len(effected.shape)) as f:
        f.write(effected)


def lowpass_filter(audio, sample_rate, directory, id):
    board = Pedalboard([
        LowpassFilter(cutoff_frequency_hz=300)
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-lpf.wav'.format(directory, id), 'w', samplerate=sample_rate,
                      channels=len(effected.shape)) as f:
        f.write(effected)


def phaser(audio, sample_rate, directory, id):
    board = Pedalboard([
        Phaser(centre_frequency_hz=1200)
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-phaser.wav'.format(directory, id), 'w', samplerate=sample_rate,
                      channels=len(effected.shape)) as f:
        f.write(effected)


def pitch_shift_up(audio, sample_rate, directory, id):
    board = Pedalboard([
        PitchShift(scale_factor=1.2)
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-psu.wav'.format(directory, id), 'w', samplerate=sample_rate,
                      channels=len(effected.shape)) as f:
        f.write(effected)


def pitch_shift_down(audio, sample_rate, directory, id):
    board = Pedalboard([
        PitchShift(scale_factor=0.8)
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-psd.wav'.format(directory, id), 'w', samplerate=sample_rate,
                      channels=len(effected.shape)) as f:
        f.write(effected)


def pitch_shift_up_up(audio, sample_rate, directory, id):
    board = Pedalboard([
        PitchShift(scale_factor=1.4)
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-psuu.wav'.format(directory, id), 'w', samplerate=sample_rate,
                      channels=len(effected.shape)) as f:
        f.write(effected)


def pitch_shift_down_down(audio, sample_rate, directory, id):
    board = Pedalboard([
        PitchShift(scale_factor=0.6)
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-psdd.wav'.format(directory, id), 'w', samplerate=sample_rate,
                      channels=len(effected.shape)) as f:
        f.write(effected)


def reverb(audio, sample_rate, directory, id):
    board = Pedalboard([
        Reverb(room_size=0.8)
    ], sample_rate=sample_rate)

    effected = board(audio)
    with sf.SoundFile('{}/{}-reverb.wav'.format(directory, id), 'w', samplerate=sample_rate,
                      channels=len(effected.shape)) as f:
        f.write(effected)


def apply_effects(directory):
    audio_file_names = Path(directory).rglob('*.wav')
    id = 1

    for audio_file_name in audio_file_names:
        audio, sample_rate = sf.read(audio_file_name)
        chorus(audio, sample_rate, directory, id)
        distortion(audio, sample_rate, directory, id)
        phaser(audio, sample_rate, directory, id)
        lowpass_filter(audio, sample_rate, directory, id)
        highpass_filter(audio, sample_rate, directory, id)
        pitch_shift_up(audio, sample_rate, directory, id)
        pitch_shift_down(audio, sample_rate, directory, id)
        pitch_shift_up_up(audio, sample_rate, directory, id)
        pitch_shift_down_down(audio, sample_rate, directory, id)
        reverb(audio, sample_rate, directory, id)

        id += 1


if __name__ == '__main__':
    apply_effects('srb-audio/dole')
    apply_effects('srb-audio/desno')
    apply_effects('srb-audio/levo')
    apply_effects('srb-audio/skoci')