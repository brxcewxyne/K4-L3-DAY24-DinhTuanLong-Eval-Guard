# Nhóm thực hiện K4 Track 3A

Tên nhóm: DinhTuanLong

Repo chung: https://github.com/brxcewxyne/K4-L3-DAY24-DinhTuanLong-Eval-Guard

Trưởng nhóm/người nộp: Đinh Tuấn Long - 2A202602620

Ngày lab/deadline áp dụng: 2026-10-12 — Asia/Ho_Chi_Minh (UTC+07:00)

Commit dùng để chấm: Không thể ghi SHA vào chính mục này — SHA của một commit chỉ được tạo ra
sau khi commit hoàn tất, nên file này không thể chứa hash của commit chứa nó; nếu nhúng SHA rồi
sửa file để thêm SHA thì hash vừa nhúng lập tức không còn khớp. Vì vậy bản nộp được xác định
bằng link repo kèm SHA gửi cho coach khi nộp theo hướng dẫn, không chèn SHA đoán trước vào đây.
Không nộp key, không nộp file .env.

Thành viên duy nhất: Đinh Tuấn Long — Mã học viên 2A202602620; thực hiện toàn bộ lab end to end.

Mỗi nhóm nộp một repo. Nhóm hiện có một thành viên thực hiện toàn bộ bài.

## Thành viên và phân công

| STT | Họ và tên | MSSV | Vai trò / công việc thật | Bằng chứng |
|---|---|---|---|---|
| 1 | Đinh Tuấn Long | 2A202602620 | M1-M5, provider OpenCode Go, Phase A/B/C, chạy live và viết báo cáo | artifacts/*.json, doc/reports/*, tests/*, working tree hiện tại (chưa commit) |

Người tích hợp/chạy batch/nộp link: Đinh Tuấn Long — chạy setup 50 answers, Phase A/B/C live và
điền các báo cáo.

## Môi trường và chuẩn bị trước lớp

Hệ điều hành/Python/compiler: Windows, Python 3.11.9; máy có MSVC Build Tools nên build được
annoy 1.17.3 cho NeMo Guardrails.

Môi trường ảo và dependencies: repo có .venv-tests (pytest + jsonschema, dùng cho unit tests
offline); môi trường chạy runtime dùng venv Python 3.11 local với đủ dependencies theo
requirements: ragas 0.1.22, nemoguardrails 0.10.0, presidio-analyzer/anonymizer 2.2.364,
spacy 3.7.6 kèm en_core_web_lg, sentence-transformers 6.1.0, qdrant-client 1.9.2, rank-bm25 0.2.2,
underthesea, pypdf, openai 1.109.1.
Lệnh kiểm tra: python -m pip check trả về No broken requirements found.

Model, Qdrant và 50 answers đã chuẩn bị: Qdrant qdrant/qdrant:v1.9.7 chạy qua
docker compose -f infra/docker-compose.yml up -d; cache sẵn BAAI/bge-m3 và BAAI/bge-reranker-v2-m3;
embedding cho evaluator dùng model local sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2;
50 answers đã sinh live bằng OpenCode Go deepseek-v4.1-flash (llm=50/50, không fallback).

Lưu ý độ phủ (coverage): hai file PDF trong data/ (BCTC.pdf và Nghị định 13/2023) là bản scan ảnh,
không có text layer nên pypdf không trích được nội dung; corpus được index còn 26/28 tài liệu.
Đã kiểm tra từ khóa trên 50 câu hỏi: không câu nào cần nội dung hai PDF này.

## INDIVIDUAL

### DinhTuanLong-2A202602620

- Phần được giao và phần thực tế đã làm: sửa lỗi encoding prefix trong M1; viết M4 (adapter RAGAS
  trả evaluator models trung thực, không thay NaN bằng 0) và M5 (enrichment dùng provider OpenCode
  Go, JSON mode thích ứng, fallback deterministic chỉ khi không có provider); thêm lớp provider
  dùng chung (llm_provider) hỗ trợ base_url, header x-opencode-session và đọc key từ biến môi
  trường; hoàn thiện Phase A tasks 1-4, Phase B tasks 5-8, Phase C tasks 9a-12; chạy setup 50
  answers và Phase A/B/C live; viết ba báo cáo và TEAM.
- Bằng chứng: artifacts/answers_50q.json, artifacts/ragas_50q.json, artifacts/judge_results.json,
  artifacts/guard_results.json; doc/reports/failure_clusters.md, bias_report.md, blueprint.md;
  các file tests/test_m1_chunking.py, test_m4_eval.py, test_m5_enrichment.py, test_llm_provider.py.
- Cách kiểm tra: chạy offline `python -m pytest tests/ -q` với .venv-tests, kết quả 122 passed,
  3 deselected; chạy live lần lượt `python scripts/setup_answers.py --live` (50 answers, 666 giây),
  `python scripts/run_experiments.py --phase a --live` (50 dòng RAGAS hợp lệ), `--phase b --live`
  (5 cặp, 10 nhãn, kappa 1.0), `--phase c --live --latency-samples 20` (attacks 20/20, clean 8/8,
  latency 20 mẫu). Kiểm tra cuối bằng `python scripts/check_lab.py`.
- Điều đã học: RAGAS faithfulness cần câu trả lời có dấu chấm kết câu (answer dạng bullet bị NaN);
  self-check của NeMo với reasoning model cần max_tokens đủ cho reasoning trước khi trả Yes/No;
  Presidio chỉ nên nạp recognizer cần thiết để tránh chặn nhầm; clean controls phát hiện đúng lỗi
  chặn-tất-cả (always-block) mà chỉ chạy attacks sẽ không thấy được.
- Giới hạn/cách xử lý: gặp NaN faithfulness ở câu 26 do định dạng bullet, đã siết prompt sinh
  answers, regenerate lại câu này từ contexts đã lưu rồi chạy lại Phase A; gặp điểm
  1.0000000000000002 do sai số float, đã clamp trong epsilon 1e-9 kèm test và vẫn raise nếu
  ngoài khoảng thật.
- Nguồn/AI hỗ trợ nếu có: có dùng trợ lý AI để hỗ trợ viết code, test và báo cáo; mọi số liệu trong
  báo cáo lấy từ artifact chạy thật và được kiểm tra bằng checker offline trước khi ghi.

## Cập nhật sau khi nộp

Chưa nộp. Bản làm việc local được giữ nguyên để trưởng nhóm rà soát rồi commit/push khi nộp;
SHA của commit nộp được gửi kèm link theo hướng dẫn nộp bài (không chèn trước vào file — xem giải
thích ở mục Commit dùng để chấm). Không có cập nhật nào sau deadline.
