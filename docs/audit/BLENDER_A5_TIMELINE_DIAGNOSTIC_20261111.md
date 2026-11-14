# Audit timeline Blender A5 — 11/11/2026

Đây là audit read-only của diagnostic, không phải benchmark retrieval, relation hay QVHighlights. Không ingest ASR. Kernel `lvhuylong/videntra-blender-a5-asrfix-diagnostic-20261111` ghi `status=success`: 2 video, 768 keyframes/vectors, 848 detections, 839 OCR observations và 48 ASR segments; các con số không chứng minh chất lượng dự đoán. GPU Tesla T4 15.636.037.632 bytes, Python 3.12.3, torch 2.8.0+cu128, worker 569,96 giây. Worker SHA-256 `61aaab695a1a6527f471c4305a67ce88d55ccaa0effb829fb00b8dead9e5831c`. Logs và outputs ở `work/kaggle/blender_a5_diagnostic_asrfix_20261111_output_v1/`.

## Kiểm tra và tái hiện

Lệnh sau chỉ đọc inputs, ghi JSON ra stdout; lưu vào một file mới nếu cần, không ghi đè artifact cũ:

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_blender_a5_timeline.py --bundle work/kaggle/blender_visual_dataset_20261110 --outputs work/kaggle/blender_a5_diagnostic_asrfix_20261111_output_v1 --ledger work/audit/blender_sequential_ledger_20261110
```

Kết quả thực tế được lưu tại `work/audit/blender_a5_timeline_audit_20261111.json`, SHA-256 `c2cc439b373a8d5129c8fed8525accab08dc3ef8b73cff860c58fca59841a432`. Audit dùng PyAV 19.0.2 trên Windows và ffprobe 8.1.2; kiểm hash bytes media so với bundle, frame đầu, tất cả audio frames, 768 keyframes đối chiếu ledger, finite vectors và khoảng ASR.

- Hai media có SHA-256 lần lượt `b2acb9bddcb384f9762f919af8f6d8b4be781e40ad2f35a37cc12beef55b9a27` và `6fc295e0e92835316ec95929ccfc664b05ed2434f673e6f5e684f98f7558a547`. Video đầu và audio đầu đều PTS=0 theo cả PyAV và ffprobe.
- Audio 48 kHz có 27.960 và 30.859 decoded frames, không thiếu PTS và không phát hiện gián đoạn; lần lượt 360 và 408 keyframes khớp ordinal, PTS và timestamp với ledger. Ma trận 768×768 hữu hạn, global index liên tục.
- 48 segments đều của Elephants Dream, khoảng 15,12–539,63 giây, có start < end và nằm trong audio. Không có segment ở Big Buck Bunny; điều này không tự chứng minh không có lời thoại.

Đối chiếu thêm bằng PyAV 16.0.1, đúng version worker, trong môi trường tạm riêng trên Windows; không thay đổi `.venv`. Lệnh kiểm tra dưới đây cài wheel PyAV vào thư mục tạm, decode audio, sau đó tự xóa thư mục đó:

```powershell
@'
import os, subprocess, sys, tempfile
with tempfile.TemporaryDirectory(prefix="videntra_av16_timeline_") as folder:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", folder, "av==16.0.1"], check=True)
    env = dict(os.environ, PYTHONPATH=folder, PYTHONDONTWRITEBYTECODE="1")
    code = '''import av, pathlib, sys
from fractions import Fraction
print("PyAV", av.__version__)
for name in ("big-buck-bunny", "elephants-dream"):
    first = last = None
    count = missing = gaps = 0
    with av.open(str(pathlib.Path(sys.argv[1]) / (name + ".mov")), metadata_errors="ignore") as c:
        for f in c.decode(audio=0):
            count += 1
            if f.pts is None or f.time_base is None or not f.sample_rate:
                missing += 1
                continue
            start = Fraction(f.pts) * f.time_base
            if first is None: first = start
            if last is not None and start != last: gaps += 1
            last = start + Fraction(f.samples, f.sample_rate)
    print(name, count, missing, gaps, first, last)
'''
    subprocess.run([sys.executable, "-c", code, "work/kaggle/blender_visual_dataset_20261110"], env=env, check=True)
'@ | .\.venv\Scripts\python.exe -B -
```

PyAV 16 cũng cho 27.960/30.859 audio frames, first PTS=0, không thiếu PTS hoặc gián đoạn. Cuối audio theo PyAV 16 là 596,48/658,325333 giây, dài hơn kết quả PyAV 19 khoảng 18,33/5,33 ms; các segment đều nằm trước phần tail này. Không suy rộng khác biệt decoder sang media khác. Kaggle đã cài PyAV 16 từ [wheel chính thức](https://pypi.org/project/av/16.0.1/) Linux CPython 3.12 sau khi kiểm SHA-256 `daecc2072b82b6a942acbdaa9a2e00c05234c61fef976b22713983c020b07992`.

Wheel [faster-whisper 1.2.1](https://pypi.org/project/faster-whisper/1.2.1/) được đối chiếu upstream có SHA-256 `79a66ad50688c0b794dd501dc340a736992a6342f7f95e5811be60b5224a26a7`: code xóa PTS khi ghép audio và có bước khôi phục timestamp sau VAD. Kernel chưa ghi hash wheel Faster-Whisper đã cài, cũng không lưu VAD chunks; kiểm tra source này không chứng minh từng timestamp được tái dựng chính xác.

`confidence` ASR hiện chứa raw average log probability, không phải calibrated probability và có thể không phù hợp schema downstream. Chưa chứng minh từ/ngữ, biên từng segment, mapping về source video, hay audio/video alignment với ground truth; cần audit độc lập trước khi ingest hoặc dùng để đánh giá. Các môi trường tạm local do lần audit tạo đã được dọn; venv `/tmp` bên trong Kaggle container không thể kiểm tra sau khi kernel kết thúc.
