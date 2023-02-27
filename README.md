# voice_phishing_ai_detector
ai 기반 보이스피싱 검출 앱을 위한 딥러닝 모델

Table of Contents
---------------------

 * Introduction
 * Environments
 * Setup

---

Introduction
------------
text detector, speech detector 2개의 모델이 있습니다.

정상 : 비정상 비율이 약 100 : 1이어서 anomaly detection(lstm autoencoder)을 이용하였습니다.

* text

![recondstruction_error_marker_x](https://user-images.githubusercontent.com/108724053/189615736-fbfabe3e-4bd7-444c-b6ad-60847bbb6ce3.png)

threshold 이상의 값을 보이스피싱으로 판별하는 모델입니다


* speech

---


Environments
------------

* text

| Python | tensorflow | pip |
|-------|--------|------|
| 3.9.12 | 2.6.0 | 22.1.2 |

---

## Setup

### Clone

- `https://github.com/yeobinhong/voice_phishing_ai_detector.git`에서 gitclone을  받습니다.

#### Windows기준
- git bash: `git pull` 실행, git repository와 본인의 local 폴더를 동기화 해줍니다.
- cmd: `python -m venv <가상 환경을 설치할 폴더 경로>` 실행, 
  'venv'라는 폴더가 생성되면 성공입니다.
- cmd: <가상 환경이 설치된 폴더>\venv\Scripts에서 `activate.bat`을 입력, 가상 환경을 실행합니다.
  cmd에서 `(venv)`를 통해 가상 환경이 활성화 된것을 확인합니다.
  <참고> Windows에는 Linux/Mac기준의 `source ./venv/bin/activate`라는 명령어가 없는 것 같습니다.
- cmd: conda_requirements.txt가 저장되어있는 폴더로 이동, `pip install -r conda_requirements.txt`실행, 가상 환경의 정보를 일치시킵니다.
- 용량이 큰 파일은 구글 드라이브에 업로드하였습니다. download_link.txt를 참고주세요.
---
