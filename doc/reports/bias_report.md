# Đánh giá LLM Judge Phase B

Nhóm: DinhTuanLong

Người phụ trách: Đinh Tuấn Long - 2A202602620

Nguồn: `artifacts/judge_results.json`; thời điểm chạy: 2026-10-12T13:38:10+00:00.
Model theo vai trò generation/judge: generation = deepseek-v4.1-flash (pipeline sinh answers);
judge = deepseek-v4.1-flash qua OpenCode Go cho cả pairwise judge và binary judge
(`experiment_support.judge_binary` dùng cùng provider/model).

## Năm cặp A/B và hai lượt

Nguồn cố định: A từ artifacts/answers_50q.json:answer; B từ
fixtures/human_labels_10q.json:model_answer; reference từ fixtures/test_set_50q.json:ground_truth.
Ids: 1, 5, 12, 21, 41. Không đổi nguồn sau khi xem kết quả judge.

| ID | Người thắng lượt 1 | Điểm lượt 1 | Người thắng lượt 2 (đã quy về A/B gốc) | Điểm lượt 2 | Kết luận cuối | Nhất quán |
|---|---|---|---|---|---|---|
| 1 | B | A 0.8 / B 0.9 | B | A 0.6 / B 0.9 | B | Có |
| 5 | A | A 1.0 / B 0.0 | A | A 1.0 / B 0.1 | A | Có |
| 12 | A | A 1.0 / B 0.7 | A | A 1.0 / B 0.7 | A | Có |
| 21 | B | A 0.5 / B 1.0 | B | A 0.5 / B 1.0 | B | Có |
| 41 | A | A 0.9 / B 0.2 | A | A 0.9 / B 0.1 | A | Có |

Nhận xét nội dung: ở cặp 1, lượt 1 khen B nêu đủ ý "có lương"; lượt 2 (đã hoán đổi) khen
ứng viên (candidate) tương ứng với A gốc là đầy đủ hơn về số ngày có lương — hai lượt đều chọn
cùng một answer thật (B), không đổi theo vị trí. Cặp 5 và 41 là câu có bẫy phiên bản: judge chọn
answer nêu đúng ngưỡng phê duyệt theo chính sách hiện hành và phiên bản v2024; answer sai dùng
phiên bản cũ bị chấm thấp. Cặp 21: hai lượt đều chọn B nhưng điểm lượt 1 thấp hơn lượt 2 một
chút, phản ánh judge có dao động nhỏ về điểm số dù quyết định không đổi.

Số cặp position bias (count/rate): 0 / 5 (0.0). Cả năm cặp giữ nguyên kết luận khi hoán đổi thứ tự.
Tỷ lệ thắng nhờ dài hơn (verbosity win rate): 0.800; số case có kết luận rõ ràng (decisive): 5.
Chi tiết: A thắng và dài hơn B 3 lần, B thắng và dài hơn A 1 lần. Đây là dấu hiệu cần đọc nội
dung trước khi kết luận: các answer dài hơn trong bộ này thường cũng đầy đủ hơn (nêu thêm điều
kiện áp dụng), nên chưa đủ căn cứ kết luận judge thiên vị độ dài; cần thêm mẫu có độ dài và chất
lượng tách rời để kiểm chứng.

## Calibration đúng model_answer

Hàm hỗ trợ (helper) đánh giá đúng question/model_answer đang có trong fixtures/human_labels_10q.json;
không lấy answer mới của pipeline. Prompt chỉ nhận question, answer, reference.
Không gửi human_label, human_note hoặc phân bố nhãn human vào prompt.

| ID | Nhãn judge | Nhãn người (human) | Đồng ý | Lý do chấm (reasoning, rút gọn) |
|---|---|---|---|---|
| 1 | 1 | 1 | Có | Đúng ý chính: 3 ngày làm việc có lương khi kết hôn |
| 5 | 0 | 0 | Có | Thiết bị 55 triệu vượt ngưỡng 50 triệu, phải do Tổng Giám đốc phê duyệt |
| 12 | 1 | 1 | Có | Đúng mức thưởng Tết tối thiểu 1 tháng lương và điều kiện 6 tháng |
| 21 | 1 | 1 | Có | Khớp v2024: 9 năm thâm niên cộng 3 ngày, tổng 18 ngày |
| 23 | 1 | 1 | Có | Đúng: nghỉ sau 8 tháng chưa đủ cam kết 1 năm nên phải hoàn trả |
| 29 | 0 | 0 | Có | Thiếu ý: 8 triệu cần cả Trưởng phòng và Kế toán trưởng phê duyệt |
| 33 | 1 | 1 | Có | Đúng: phụ cấp Manager 1.500.000 đồng/tháng và 19 ngày phép |
| 41 | 0 | 0 | Có | Sai: dùng chính sách cũ v2023 (12 ngày) thay vì bản hiện hành |
| 46 | 1 | 1 | Có | Đúng phủ định trọng tâm: nhân viên thử việc không được nghỉ phép năm |
| 50 | 0 | 0 | Có | Sai: chính sách VPN cấm dùng VPN cá nhân, phải dùng WireGuard của công ty |

Cohen kappa: 1.000. Đồng thuận quan sát (observed agreement) = 10/10 = 1.0. Đồng thuận ngẫu
nhiên (chance agreement) = (6/10 × 6/10) + (4/10 × 4/10) = 0.52. Kappa = (1.0 − 0.52) / (1 − 0.52)
= 1.0. Với 10 nhãn (6 nhãn 1 và 4 nhãn 0 trong dữ liệu human), mức đồng thuận này rất cao; xem
diễn giải bên dưới.

## Giới hạn và cách dùng judge

- Mẫu nhỏ: 5 cặp, 10 nhãn nhị phân (binary); kappa 1.0 không đảm bảo tổng quát hóa cho tập lớn
  hơn. Cần thêm mẫu và nhãn người trước khi dùng judge làm cổng chặn tự động.
- Cả hai quy trình judge (pairwise và gán nhãn 0/1) dùng cùng model deepseek-v4.1-flash; sai
  lệch hệ thống của model sẽ ảnh hưởng đồng thời cả hai quy trình, không có nguồn đối chứng độc
  lập trong lượt chạy này.
- Nhãn người (human labels) chỉ được dùng sau khi judge chấm xong, đúng thứ tự id; kết quả đồng
  ý tuyệt đối có thể một phần do các câu được chọn đã tương đối rõ ràng và model_answer cung cấp
  có độ chênh lệch lớn về chất lượng.
- Judge không phải đáp án tuyệt đối: khi dùng cho CI nên đặt ngưỡng đồng thuận tối thiểu, lấy
  mẫu người kiểm tra định kỳ và lưu lý do chấm để kiểm toán (audit). Với các câu đánh giá cao
  nhưng lý do ngắn, nên đọc kỹ nội dung trước khi tin.
