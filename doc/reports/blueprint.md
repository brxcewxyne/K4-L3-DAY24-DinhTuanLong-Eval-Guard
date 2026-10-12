# Blueprint CI/CD cho RAG và guardrails

Nhóm: DinhTuanLong

Người phụ trách: Đinh Tuấn Long - 2A202602620

Đây là **đề xuất CI/CD**, không phải một quy trình đã được triển khai hay chạy thực tế.
Nguồn kết quả: artifacts/ragas_50q.json, judge_results.json, guard_results.json.
Thời điểm chạy: guard 2026-10-12T13:44:36+00:00; model guard: deepseek-v4.1-flash qua OpenCode Go.
Model theo vai trò generation/evaluator/judge: generation = deepseek-v4.1-flash;
evaluator = deepseek-v4.1-flash + sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(embeddings local); judge = deepseek-v4.1-flash.

## Kiến trúc và hành động khi lỗi (failure action)

Luồng đề xuất: Input → Presidio → NeMo input rail → RAG (hybrid BM25 + dense, cross-encoder
rerank) → NeMo output rail → phản hồi. Request thật dừng sớm khi bị chặn (short-circuit), khác
với phép đo Task 12 luôn chạy đủ hai tầng.

Hành động khi lỗi theo từng lớp:
- Presidio phát hiện PII: chặn ngay, ghi `blocked_by = presidio`, ghi log loại entity và vị trí;
  không đưa nội dung PII vào context. Có thể chọn thay thế/che thông tin (redact) thay vì chặn
  tùy luồng nghiệp vụ.
- NeMo input rail: chặn câu hỏi ngoài phạm vi, jailbreak, prompt injection; trả câu từ chối
  chuẩn trong rails.co; ghi `blocked_by = nemo_input`.
- RAG: nếu không có context, pipeline hiện trả câu "Không tìm thấy thông tin." ở chế độ có kiểm
  soát; trong chế độ live strict, lỗi retrieval làm lần chạy thất bại thay vì tạo câu trả lời rỗng.
- NeMo output rail: output nhạy cảm bị thay bằng câu từ chối, output sạch giữ nguyên; 4/4 case
  trong fixture đều đúng (2 sạch giữ nguyên, 2 nhạy cảm bị chặn).
- Bí mật (secrets): chỉ đọc từ biến môi trường/.env đã gitignore; log không in key; không commit key.

## Phạm vi và số đo latency

Bắt buộc Task 12: **input_layers_sequential**, Presidio + NeMo input, không short-circuit.
Các engine được tạo trước khi đo; có hai lượt warm-up thật không tính vào mẫu. Percentile được
nội suy tuyến tính từ các mẫu thô (raw samples); input_guard_ms là tổng của từng sample.

Số mẫu đo (timed): 20; warm-up: 2; phương pháp: linear.

| Lớp / phạm vi | P50 ms | P95 ms | P99 ms | Tình trạng |
|---|---|---|---|---|
| Presidio | 8.28 | 13.61 | 14.45 | Đã đo Task 12 |
| NeMo input (deepseek-v4.1-flash) | 2343.28 | 2712.34 | 2849.32 | Đã đo Task 12 |
| Input guard sequential | 2353.24 | 2720.15 | 2857.28 | Đã đo Task 12 |
| RAG | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| NeMo output | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| Full guard / request short-circuit | chưa đo | chưa đo | chưa đo | Không suy từ input_guard_ms |

Mục tiêu ngân sách cho tầng input: 500ms; kết quả thực tế: budget_ok = false, P95 input guard
2720.15 ms (vượt khoảng 5.4 lần). Nút thắt (bottleneck) là NeMo self-check: deepseek-v4.1-flash
là reasoning model, mỗi lượt tự kiểm tốn khoảng 2.3 giây ở trung vị dù chỉ cần trả lời Yes/No.
Đề xuất: cấu hình max_tokens vừa đủ (đang dùng 512 để reasoning hoàn tất trước khi trả lời),
giảm số lượt self-check cho câu ngắn, hoặc dùng model chat nhỏ hơn cho riêng rail self-check;
sau đó đo lại và đặt budget theo số thật. Không dùng input_guard_ms để kết luận total guard đạt
500ms.

## Cổng kiểm tra CI (CI gates) đề xuất

- RAGAS: faithfulness ≥ 0.75 và avg_score ≥ 0.65 trên 50 câu. Số hiện tại: faithfulness tổng
  khoảng 0.904, avg_score tổng khoảng 0.796; riêng multi_hop avg 0.678 sát ngưỡng, nên thêm cổng
  riêng "multi_hop avg ≥ 0.65" để bắt sớm suy giảm truy xuất nhiều tài liệu.
- Guard: attacks ≥ 18/20 cho cổng CI đề xuất (hiện tại 20/20, trong đó 4 chặn bởi Presidio và 16
  bởi NeMo input); clean 8/8 allowed, false_positive_rate = 0; output 4/4 đúng. Mốc bắt buộc của
  lab vẫn là attacks ≥ 15/20 kèm clean controls đạt.
- Judge: chạy hiệu chuẩn (calibration) với human labels khi có mẫu mới; hiện κ = 1.0 trên 10 nhãn,
  chỉ dùng làm cảnh báo chứ chưa phải cổng chặn cứng.
- Latency: cổng full guard chỉ áp dụng sau khi thực sự đo full guard; hiện tại chỉ có số đo ở
  scope input layers. Đề xuất ghi "chưa đo" thay vì tuyên bố đạt.
- Quy trình CI chưa được triển khai: đề xuất một bước CI chạy `python scripts/check_lab.py --with-tests`
  sau khi sinh artifacts và đánh dấu lần build thất bại khi checker trả mã khác 0; các ngưỡng MIN_* là
  biến đề xuất, không tự động thành quality gate nếu runner chưa kiểm tra ngưỡng.

## Giám sát (monitoring) và cải tiến

- Theo dõi mỗi lượt chạy: avg_score theo distribution, các câu điểm thấp nhất, tỷ lệ chặn theo
  lớp (presidio/nemo), false_positive_rate trên clean controls, P95 input guard; lưu mẫu thô
  (raw samples) và SHA-256 của các input để phát hiện trôi dữ liệu.
- Ngưỡng cảnh báo đề xuất: multi_hop avg giảm dưới 0.65; bất kỳ clean control nào bị chặn; output
  sạch bị thay đổi; P95 input guard tăng quá 20% so với mức cơ sở (baseline) gần nhất.
- Lấy mẫu: giữ 20 mẫu latency sau 2 warm-up mỗi lần đo; lấy thêm mẫu câu hỏi thật (ẩn danh) để
  bổ sung bộ đánh giá; rà soát thủ công 10 câu điểm thấp nhất mỗi chu kỳ.
- Xử lý false positive: mọi false positive từ clean controls phải mở issue và bổ sung vào bộ
  clean set trước khi tăng độ nhạy của guard.
- Trôi dữ liệu (drift) và quyền riêng tư (privacy): theo dõi phiên bản tài liệu (v2023/v2024)
  khi corpus đổi; không log nội dung PII, chỉ log loại entity và vị trí; người xử lý: nhóm trưởng
  kiêm trực giám sát trong lab.

RAGAS: điểm trung bình (mean) 0.796; nhóm yếu nhất theo mean là multi_hop với 0.678.
Cohen kappa: 1.000 (10 nhãn, xem giới hạn trong bias_report).
Số attacks đạt: 20/20; false_positive_rate trên clean: 0.0.

Đề xuất cải tiến gắn bằng chứng, có thể kiểm tra lại:
1. Nâng context cho câu multi-hop (top-k lớn hơn hoặc gộp parent chunk) để cải thiện
   context_recall multi_hop 0.579, đo lại bằng Phase A.
2. Siết prompt generation trả lời đủ mọi vế của câu hỏi nhiều ý nhằm nâng answer_relevancy
   multi_hop 0.566; đối chiếu bottom 10 trước và sau.
3. Giảm latency self-check bằng model nhỏ hơn hoặc cache cho các câu hỏi lặp; đo lại P95 input
   guard (hiện 2720 ms) và chỉ đặt cổng cho full guard sau khi có đủ số đo.
4. Mở rộng hiệu chuẩn judge lên tối thiểu 30 cặp trước khi dùng làm cổng tự động, vì κ hiện tại
   chỉ dựa trên 10 nhãn.
5. Bổ sung pipeline OCR cho hai PDF scan trong data/ nếu bổ sung câu hỏi liên quan; hiện corpus
   được index 26/28 tài liệu và 50 câu hỏi không phụ thuộc hai PDF này.
