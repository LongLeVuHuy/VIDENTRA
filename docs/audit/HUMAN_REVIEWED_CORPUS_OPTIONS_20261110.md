# Nguồn corpus và nhãn người — khảo sát 10/11/2026

Mục tiêu là tìm video có điều kiện sử dụng rõ, nhãn thời gian do người rà soát và nguồn tách độc lập. Các kết luận dưới đây chỉ dựa vào thông tin nhà cung cấp công bố, không xác nhận quyền cho asset chưa kiểm hay khả năng chuyển dữ liệu lên Kaggle.

| Nguồn | Bằng chứng hữu ích | Giới hạn trước khi dùng |
| --- | --- | --- |
| [TRECVID-AVS — features và judgments trên Zenodo](https://zenodo.org/records/13941108) | Nhà phát hành mô tả 89 query với shot được người gán relevant/non-relevant; họ chọn 80 bộ, mỗi bộ 1.500 keyframe, phát hành đặc trưng CLIP ViT-H/14 và judgment dưới Creative Commons Attribution. | Chỉ là đặc trưng/frame và nhãn shot; pool cho từng query đã được chọn theo nhãn (10% positive), không có raw media/ledger, không thể coi là full-corpus moment benchmark, full A5 hay so trực tiếp SigLIP. File normalized khoảng 1,1 GB; cần kiểm license cụ thể, protocol và exposure trước dùng. |
| [V3C/TRECVID data](https://www-nlpir.nist.gov/projects/tv2024/data.html) | NIST mô tả V3C1 là Vimeo Creative Commons Collection dùng cho ad-hoc video search. | Cần đăng ký TRECVID và permission forms để nhận dữ liệu; điều kiện từng asset, sử dụng ngoài hạ tầng và Kaggle cần kiểm riêng. Không tự cho rằng metadata CC bao phủ mọi thành phần. |
| [MicroAnnotationBench](https://huggingface.co/datasets/MicroAGI-Labs/MicroAnnotationBench) | Nhà cung cấp mô tả 45 video phát hành, timeline hành động do người gán và split theo operator/session; nhãn và data công bố CC BY-NC 4.0. | Gated, yêu cầu nộp form và được chấp thuận; không redistribute, không train; 5 GB. Task là action segmentation, không trực tiếp natural-language moment retrieval. Việc upload private Kaggle cần kiểm điều khoản, không tự suy ra. |
| [Cultural Moment Benchmark](https://culturalmoment-benchmark.github.io/) | Benchmark test-only cho Đông Nam Á, annotation và evaluation theo CC BY-NC-SA 4.0. | Video chỉ được tham chiếu bằng YouTube ID, copyright ở uploader; không mặc định có quyền media hoặc upload Kaggle. Cần kiểm thêm provenance nhãn và mapping timeline từng clip. |
| [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) | Dataset card nêu nguồn video YouTube CC-BY, credit cho kênh và transcript theo video. | Metadata thời gian do Gemini/GPT tạo; không mặc định là human-reviewed moment gold. Cần rà soát license/tính toàn vẹn từng video và human review độc lập. |
| [MomentBench](https://github.com/mixpeek/momentbench) | Có media source/attribution, các query và moment với hard negative. | Nhà cung cấp ghi rõ nhãn do model tạo, model adjudicate và chưa human review; pilot Blender hiện tại chỉ mang tính thăm dò. |

## Hướng thực hiện

1. Không dùng gold để chọn candidate, frame hoặc rerank. Khóa universe theo source và quyền trước, sau đó evaluator mới join nhãn và báo coverage kể cả query không có target trong index.
2. Dùng hai phim Blender có receipt hiện có làm pilot kỹ thuật, nhưng không ghi nhãn do model tạo là human truth. Muốn đánh giá đáng tin cậy cần human reviewers độc lập, nhãn khoảng thời gian/quan hệ, agreement và xét bất đồng; giữ nguyên bản gốc và lịch sử.
3. Nếu có quyền tham gia TRECVID hoặc MicroAnnotationBench, kiểm agreement, phạm vi license và tính phù hợp của protocol trước khi ingest; không mặc định upload dữ liệu restricted lên Kaggle.
4. Có thể xem Zenodo features như benchmark **CLIP frame-only** riêng với declared label-selected candidate pools; không báo là kết quả VIDENTRA A5 hoặc video-moment retrieval trên corpus độc lập.
