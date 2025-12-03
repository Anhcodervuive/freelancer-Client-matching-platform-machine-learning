# Vì sao lấy trung bình embedding theo từng skill lại hiệu quả hơn?

1. **Giảm nhiễu do định dạng chuỗi**
   - Chuỗi dài kiểu `"React, NodeJS, CSS"` là một câu giả không có ngữ pháp; mô hình câu phải đoán ngữ cảnh nên dễ lệch hướng.
   - Khi embed từng skill rồi lấy mean, mỗi token đã được mô hình tối ưu (đây là tên thực thể/kỹ năng), tránh việc mô hình phải hiểu dấu phẩy hay thứ tự.

2. **Bớt nhạy thứ tự/alias**
   - Danh sách ghép chuỗi phụ thuộc vị trí: đổi vị trí kỹ năng có thể làm cosine thay đổi.
   - Mean pooling sau chuẩn hoá và dedup giúp vector đại diện ổn định hơn: hoán đổi, thêm/bớt trùng lặp không làm vector lệch nhiều.

3. **Giữ độ lớn vector cho cosine**
   - Nếu embed cả chuỗi rồi chuẩn hoá, khi tách từng skill sẽ nhỏ lẻ và khó cộng gộp.
   - Quy trình hiện tại: (a) chuẩn hoá từng vector skill, (b) lấy trung bình, (c) chuẩn hoá lại. Điều này đảm bảo độ dài L2 ≈ 1, cosine phản ánh góc chứ không bị giảm vì vector ngắn lại sau khi cộng.

4. **Tương thích với các feature rời**
   - Dễ kết hợp với số liệu overlap/Jaccard: cùng một danh sách skill đã normalize/dedup có thể dùng cho cả embedding mean và tính coverage.
   - Một pipeline duy nhất vừa phục vụ cosine, vừa phục vụ thống kê đếm trùng.

5. **Mở rộng được cho weighting**
   - Nếu muốn ưu tiên kỹ năng chính, có thể gán trọng số khi tính trung bình (weighted mean) mà không cần đổi schema DB.
   - Hướng này cũng cho phép thử nghiệm tăng trọng số cho các skill domain so với tooling phụ.

Tóm lại, mean pooling trên từng skill (kèm normalize + renormalize) giúp vector đại diện ổn định, ít phụ thuộc format, và vẫn giữ được độ sắc nét của cosine khi so khớp freelancer–job.
