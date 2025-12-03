# Mean pooling là gì và vì sao hiệu quả hơn khi so khớp skill?

## Mean pooling là gì?
- Embed từng skill riêng lẻ → thu được nhiều vector đã chuẩn hoá (L2 = 1 nếu dùng `normalize_embeddings=True`).
- Cộng các vector lại rồi chia cho số lượng (tức lấy trung bình từng chiều). Đây chính là **mean pooling**.
- Tuỳ chọn `renormalize_output=True` giúp chuẩn hoá lại vector trung bình (độ dài L2 ≈ 1) để cosine chỉ đo góc, không bị giảm vì độ dài nhỏ đi sau khi lấy trung bình.

Ví dụ: hai skill vuông góc có embedding `[1, 0]` và `[0, 1]`
- Mean: `[0.5, 0.5]` (độ dài ~0.707, cosine sẽ thấp nếu không renorm)
- Renorm: `[0.707, 0.707]` (độ dài 1, cosine giữ được độ sắc nét)

## Vì sao mean pooling ổn định hơn embedding cả chuỗi
1. **Giảm nhiễu do định dạng chuỗi**
   - Chuỗi dài kiểu `"React, NodeJS, CSS"` là một câu giả không có ngữ pháp; model phải đoán ngữ cảnh nên dễ lệch.
   - Embed từng skill rồi mean giúp model nhìn từng thực thể độc lập, không bị ảnh hưởng bởi dấu phẩy/thứ tự.

2. **Bớt nhạy thứ tự/alias**
   - Ghép chuỗi phụ thuộc vị trí, đổi chỗ skill có thể làm cosine thay đổi.
   - Normalize + dedup + sort trước khi embed rồi mean → hoán đổi/thêm bớt trùng lặp gần như không làm vector lệch.

3. **Giữ độ lớn vector cho cosine**
   - Với pipeline hiện tại: chuẩn hoá từng vector → mean → (tuỳ chọn) chuẩn hoá lại. Cosine phản ánh đúng góc giữa hai tập kỹ năng, không bị “mềm” đi vì vector ngắn dần khi cộng nhiều skill.

4. **Tương thích với feature overlap**
   - Cùng danh sách skill đã normalize/dedup dùng cho cả mean pooling lẫn các metric overlap (Jaccard, coverage), giúp score tổng hợp nhất quán.

5. **Linh hoạt cho weighting**
   - Có thể mở rộng sang weighted mean (gán trọng số cho skill quan trọng) mà không đổi schema DB; chỉ cần điều chỉnh khi tính toán.

> Tóm lại, mean pooling trên từng skill (kèm normalize + renormalize) tạo vector đại diện ổn định, ít phụ thuộc format, và giữ độ sắc nét của cosine khi so khớp freelancer–job.
