#  # D I N A --------------
# numpy_array = np.frombuffer(audio.get_wav_data(), dtype=np.float32)
# print(numpy_array)
# new_array = np.ndarray(shape=numpy_array.shape)
#
# for i in range(len(numpy_array)):
#     if str(numpy_array[i]) == "nan":
#         new_array[i] = 0;
#     else:
#         new_array[i] = numpy_array[i]
#
# print(new_array)
# return
#
# mel_spectrogram = librosa.feature.melspectrogram(new_array, sr=audio.sample_rate, n_fft=2048, n_mels=128)
# log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
#
# px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
# plt.figure(figsize=(42 * px, 42 * px))
# plt.axis('off')
# librosa.display.specshow(log_mel_spectrogram, sr=audio.sample_rate)
# plt.set_cmap('magma')
# plt.savefig('microphone-results.png', bbox_inches='tight',
#             transparent=True, pad_inches=0.0)
#
# image = np.array(Image.open('microphone-results.png'))
# pred = MODEL.predict(np.array([image]))
# detected_command = COMMANDS[np.argmax(pred, axis=1)[0]]
# print("-------------------------------------------------")
# print(detected_command)

# dina //