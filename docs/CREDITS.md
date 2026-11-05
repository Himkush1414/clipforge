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
| [OpenCV](https://opencv.org/) | Face detection for reframing | Apache 2.0 |
| [FastAPI](https://fastapi.tiangolo.com/), [SQLModel](https://sqlmodel.tiangolo.com/) | Local API and database | MIT |
| [Next.js](https://nextjs.org/), [React](https://react.dev/), [Tailwind CSS](https://tailwindcss.com/) | Web UI | MIT |

## Workflow inspiration

These projects' public docs influenced how clipforge was built. No code from them is in this repo, and none of them were installed.

- [ponytail](https://github.com/DietrichGebert/ponytail): small complete changes, with a test for new logic
- [ECC](https://github.com/affaan-m/ECC): the verification loop and FastAPI layout
- [gstack](https://github.com/garrytan/gstack): the pre-merge review checklist and browser QA
- [ruflo](https://github.com/ruvnet/ruflo): recording decisions with their reasons
