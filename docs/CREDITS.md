# Credits

## Fonts

Bundled in `engine/clipforge/assets/fonts/`. All are under the [SIL Open Font License 1.1](https://openfontlicense.org/), and each licence file sits beside its font. The fonts come from the [Google Fonts repository](https://github.com/google/fonts).

| Font | Copyright | Licence file |
|---|---|---|
| Anton | 2020 The Anton Project Authors | `anton.OFL.txt` |
| Archivo Black | 2017 The Archivo Black Project Authors | `archivoblack.OFL.txt` |
| Bangers | 2010 The Bangers Project Authors | `bangers.OFL.txt` |
| Bebas Neue | 2010 by Dharma Type | `bebasneue.OFL.txt` |
| Hind (Bold) | 2014, Indian Type Foundry (info@indiantypefoundry.com) | `hind.OFL.txt` |
| Noto Sans Arabic (Bold) | 2022 The Noto Project Authors | `notosansarabic.OFL.txt` |
| Noto Sans Hebrew (Bold) | 2022 The Noto Project Authors | `notosanshebrew.OFL.txt` |
| Lilita One | 2011 Juan Montoreano (juan@remolacha.biz) | `lilitaone.OFL.txt` |
| Poppins (Black, ExtraBold) | 2020 The Poppins Project Authors | `poppins.OFL.txt` |

Downloaded by `scripts/setup.sh`, not stored in this repo:

| Font | Copyright | Licence |
|---|---|---|
| [Noto Sans CJK](https://github.com/notofonts/noto-cjk) Bold | 2014-2021 Adobe | [SIL Open Font License 1.1](https://openfontlicense.org/) |

## Software used at runtime

These are separate programs and libraries, not copied into this repo.

| Project | Use | Licence |
|---|---|---|
| [FFmpeg](https://ffmpeg.org/) | Decoding, filters, encoding. The static build used in setup is GPL. | LGPL/GPL |
| [libass](https://github.com/libass/libass) | Caption rendering | ISC |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Transcription | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | Whisper inference | MIT |
| [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | English word timing model, downloaded by setup as the [Xenova ONNX export](https://huggingface.co/Xenova/wav2vec2-base-960h) | Apache 2.0 |
| [wav2vec2-large-xlsr-53-spanish](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish) | Spanish word timing model, downloaded by setup as an [ONNX export](https://huggingface.co/XDcobra/wav2vec2-large-xlsr-53-spanish-ONNX) | Apache 2.0 |
| [wav2vec2-base-arabic](https://huggingface.co/onnx-community/wav2vec2-base-arabic-without-reused-en-token-ids-v1-ONNX) | Arabic word timing model, downloaded by setup | Apache 2.0 |
| [IndicWav2Vec Hindi](https://huggingface.co/ai4bharat/indicwav2vec-hindi) | Hindi word timing model, downloaded by setup as an [ONNX export](https://huggingface.co/onnx-community/indicwav2vec-hindi-ONNX) | Apache 2.0 |
| [ONNX Runtime](https://onnxruntime.ai/) | Runs the word timing models | MIT |
| [OpenCV](https://opencv.org/) | Runs the face detectors for reframing | Apache 2.0 |
| [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Face detection model, downloaded by setup | MIT |
| [FastAPI](https://fastapi.tiangolo.com/), [SQLModel](https://sqlmodel.tiangolo.com/) | Local API and database | MIT |
| [Next.js](https://nextjs.org/), [React](https://react.dev/), [Tailwind CSS](https://tailwindcss.com/) | Web UI | MIT |

## Workflow inspiration

These projects' public docs influenced how clipforge was built. No code from them is in this repo, and none of them were installed.

- [ponytail](https://github.com/DietrichGebert/ponytail): small complete changes, with a test for new logic
- [ECC](https://github.com/affaan-m/ECC): the verification loop and FastAPI layout
- [gstack](https://github.com/garrytan/gstack): the pre-merge review checklist and browser QA
- [ruflo](https://github.com/ruvnet/ruflo): recording decisions with their reasons
