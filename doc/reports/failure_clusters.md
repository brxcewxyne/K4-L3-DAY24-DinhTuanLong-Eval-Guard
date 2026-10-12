# Phân tích lỗi Phase A

Nhóm: DinhTuanLong

Người phụ trách: Đinh Tuấn Long - 2A202602620

Nguồn: `artifacts/ragas_50q.json`; thời điểm chạy: 2026-10-12T13:36:11+00:00.
Chế độ sinh (generation_mode): llm (50/50) qua OpenCode Go, model `deepseek-v4.1-flash`.
Model theo vai trò generation/evaluator: generation = deepseek-v4.1-flash; evaluator =
deepseek-v4.1-flash (chat) + `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(embeddings chạy local, vì OpenCode Go không cung cấp embeddings endpoint).

## Điểm theo distribution

| Chỉ số (metric) | factual (20) | multi_hop (20) | adversarial (10) |
|---|---|---|---|
| faithfulness | 1.000 | 0.771 | 0.980 |
| answer_relevancy | 0.914 | 0.566 | 0.832 |
| context_precision | 0.929 | 0.796 | 1.000 |
| context_recall | 0.700 | 0.579 | 0.600 |
| avg_score | 0.886 | 0.678 | 0.853 |

## Top 10 câu điểm thấp nhất (bottom 10) và nguyên nhân

10 dòng từ Task 3 (các khóa rank, id, distribution, avg_score, worst_metric, diagnosis,
suggested_fix). Cột diagnosis và suggested_fix giữ nguyên văn chuỗi do artifact xuất ra từ
DIAGNOSTIC_TREE của bài; phần diễn giải tiếng Việt nằm ở bảng ngay dưới.

| Hạng | ID | Nhóm | avg_score | Metric yếu nhất | diagnosis (nguyên văn) | suggested_fix (nguyên văn) |
|---|---|---|---|---|---|---|
| 1 | 39 | multi_hop | 0.2500 | answer_relevancy | Answer doesn't match question | Improve prompt template |
| 2 | 35 | multi_hop | 0.2917 | answer_relevancy | Answer doesn't match question | Improve prompt template |
| 3 | 30 | multi_hop | 0.3750 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 4 | 33 | multi_hop | 0.5000 | answer_relevancy | Answer doesn't match question | Improve prompt template |
| 5 | 38 | multi_hop | 0.6083 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 6 | 37 | multi_hop | 0.6138 | context_recall | Missing relevant chunks | Improve chunking or add BM25 |
| 7 | 21 | multi_hop | 0.6250 | answer_relevancy | Answer doesn't match question | Improve prompt template |
| 8 | 27 | multi_hop | 0.6250 | answer_relevancy | Answer doesn't match question | Improve prompt template |
| 9 | 40 | multi_hop | 0.6421 | context_recall | Missing relevant chunks | Improve chunking or add BM25 |
| 10 | 23 | multi_hop | 0.6880 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |

Diễn giải tiếng Việt cho các giá trị diagnosis/suggested_fix xuất hiện trong bảng trên:

| diagnosis (nguyên văn) | Nghĩa tiếng Việt | suggested_fix (nguyên văn) | Nghĩa tiếng Việt |
|---|---|---|---|
| LLM hallucinating | Mô hình sinh nội dung không có căn cứ trong ngữ cảnh (ảo giác) | Tighten system prompt, lower temperature | Siết prompt hệ thống, giảm temperature |
| Missing relevant chunks | Thiếu các đoạn ngữ cảnh liên quan | Improve chunking or add BM25 | Cải thiện chunking hoặc bổ sung BM25 |
| Answer doesn't match question | Câu trả lời không khớp câu hỏi | Improve prompt template | Cải thiện prompt template |
| Too many irrelevant chunks | Quá nhiều đoạn ngữ cảnh không liên quan | Add reranking or metadata filter | Thêm reranking hoặc lọc theo metadata |

Câu hỏi tương ứng (rút gọn):

| ID | Câu hỏi (rút gọn) |
|---|---|
| 39 | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA |
| 35 | Nhân viên Junior P1 lương cơ bản 12 triệu vừa vào thử việc: nhận lương bao nhiêu và phụ cấp gì |
| 30 | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và nhân viên chính thức |
| 33 | Nhân viên Manager thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 |
| 38 | Công tác nước ngoài 4 ngày ở thành phố lớn, khách sạn 200 USD/đêm 3 đêm: chi phí được duyệt |
| 37 | Tự ý xóa malware và chia sẻ thông tin sự cố trên Slack nội bộ vi phạm chính sách nào |
| 21 | Nhân viên Senior 9 năm thâm niên: số ngày phép năm và khoảng lương trong kỳ đánh giá |
| 27 | Thâm niên 7 năm theo v2024: bao nhiêu ngày phép, nếu bị trừ thêm 4 ngày thì còn lại bao nhiêu |
| 40 | Manager thử việc 90 ngày, tháng thứ 3, phát hiện vi phạm bảo mật: quy trình và quyền hạn xử lý |
| 23 | Được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học: phải hoàn trả bao nhiêu |

Đối chiếu answer/contexts/reference cho thấy nhóm lỗi chính là **câu multi-hop so sánh nhiều
nguồn** (so sánh phiên bản v2023/v2024, thử việc/chính thức, nhiều bước tính toán). Lỗi
answer_relevancy chiếm 5/10 dòng bottom: câu trả lời thường chỉ trả lời một phần của câu hỏi
nhiều ý, nên điểm đối chiếu với question bị thấp dù nội dung không hoàn toàn sai. Lỗi
context_recall (2 dòng) đến từ việc truy xuất thiếu một trong các tài liệu cần ghép (ví dụ
chính sách v2023 và v2024, hoặc quy trình xử lý sự cố kèm phân loại dữ liệu).

## Ma trận matrix và nhóm yếu nhất

Ma trận 4 metric × 3 distribution, mỗi câu được đếm theo worst_metric của câu đó:

| Metric yếu nhất | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0 | 5 | 0 |
| answer_relevancy | 9 | 7 | 3 |
| context_precision | 4 | 2 | 0 |
| context_recall | 7 | 6 | 7 |

Nhóm yếu nhất theo điểm trung bình (mean) thấp nhất: multi_hop; giá trị mean: 0.678.
Metric yếu nhất xuất hiện nhiều nhất: context_recall; số câu: 20/50.

Giải thích: multi_hop yếu nhất vì mỗi câu cần ghép ít nhất hai mảnh chính sách (nhiều tài liệu
khác nhau) và bắt buộc dùng đúng phiên bản; retrieval top-3 đôi khi thiếu một tài liệu, đồng
thời answer dài nhiều ý làm answer_relevancy giảm. Hai metric có tổng lớn nhất là context_recall
(20) và answer_relevancy (19), nhưng tổng cột chỉ phản ánh điểm yếu nhất của từng câu, không
phải số lỗi độc lập. Đề xuất sửa cụ thể: tăng ngân sách ngữ cảnh cho câu multi-hop (top-k lớn hơn
hoặc gộp context theo parent), cải thiện prompt generation để trả lời đủ mọi vế của câu hỏi và
nêu rõ phiên bản áp dụng; cách chạy lại để xác nhận: chạy lại `python scripts/setup_answers.py --live`
và `python scripts/run_experiments.py --phase a --live`, so sánh bottom_10 và per_distribution mới.

## Bẫy adversarial và giới hạn

Adversarial (10 câu bẫy phiên bản/phủ định) có điểm trung bình 0.853, cao hơn multi_hop nhưng
vẫn thấp hơn factual. Top 10 câu điểm thấp nhất chỉ có 1 câu adversarial; các câu 41, 46, 50
không nằm trong danh sách này, cho thấy điểm thấp hơn **không tự chứng minh** pipeline đã nhận
diện đúng bẫy; cần đọc nội dung từng câu. Ví dụ answer cho câu hỏi phiên bản có lúc trích đúng
v2024 (pair 41 của Phase B cho thấy model answer cũ dùng 12 ngày v2023 bị judge chấm nhãn 0).

Giới hạn: bộ 50 câu còn nhỏ; evaluator là LLM (deepseek-v4.1-flash) kết hợp embeddings chạy
local nên có thể còn sai lệch; mỗi metric chỉ chạy một lượt, không lặp để đo dao động. Hai tài
liệu PDF trong `data/` là bản scan không có text layer, không trích được nội dung (corpus được
index còn 26/28 tài liệu); kiểm tra từ khóa cho thấy không có câu hỏi nào trong test set cần nội
dung hai PDF này, nhưng đây vẫn là khoảng trống độ phủ (coverage) cần ghi nhận.
