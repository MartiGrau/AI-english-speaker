
SST_TYPE = "whisper" # ["deepspeech", "whisper"]

# DeepSpeech
MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
MODEL_LOCAL_PATH = "/hdd1/Checkpoints/StageInHome/speech_models/deep_speech/deepspeech-0.9.3-models.pbmm"
LANG_MODEL_LOCAL_PATH = "/hdd1/Checkpoints/StageInHome/speech_models/deep_speech/deepspeech-0.9.3-models.scorer"
lm_alpha = 0.931289039105002
lm_beta = 1.1834137581510284
beam = 100

# Whisper
WHISPER_MODEL = "openai/whisper-large-v3"