# Mean pooling là gì và vì sao hiệu quả hơn khi so khớp skill?

## Mean pooling là gì?
- Embed **từng skill** riêng lẻ → thu được nhiều vector đã chuẩn hoá (L2 = 1 nếu dùng `normalize_embeddings=True`).
- Cộng các vector lại rồi chia cho số lượng (tức lấy trung bình từng chiều). Đây chính là **mean pooling**.
- Tuỳ chọn `renormalize_output=True` giúp chuẩn hoá lại vector trung bình (độ dài L2 ≈ 1) để cosine chỉ đo góc, không bị giảm vì độ dài nhỏ đi sau khi lấy trung bình.

Ví dụ: hai skill vuông góc có embedding `[1, 0]` và `[0, 1]`
- Mean: `[0.5, 0.5]` (độ dài ~0.707, cosine sẽ thấp nếu không renorm)
- Renorm: `[0.707, 0.707]` (độ dài 1, cosine giữ được độ sắc nét)

## So sánh với cách ghép chuỗi skill
| Tình huống | Ghép chuỗi thành một câu | Mean pooling trên từng skill |
| --- | --- | --- |
| Stack giống nhau nhưng thứ tự khác | Cosine có thể lệch vì model coi đây là hai câu khác nhau | Giống nhau vì danh sách đã normalize + sort rồi mới lấy trung bình |
| Alias/viết khác nhau ("Node.js" vs "NodeJS") | Model phải tự suy diễn, dễ bị nhiễu bởi dấu chấm, viết hoa | Normalize alias trước, embedding nhận cùng một nội dung nên mean giống hệt |
| Danh sách dài kèm skill phụ | Mô hình câu bị kéo về kỹ năng phụ (context trộn) | Mỗi skill đóng góp một vector; kỹ năng phụ chỉ làm mean dịch nhẹ, không lấn át hoàn toàn |
| Cần weighting (skill quan trọng hơn) | Khó chèn trọng số vào một câu dài | Có thể nhân trọng số từng vector trước khi cộng rồi chia lại |

### Ví dụ số cụ thể
- Job: `["Node.js", "React", "TypeScript", "REST API"]`
- Freelancer A: `["NodeJS", "ReactJS", "TypeScript", "REST APIs"]`
- Freelancer B: `["NodeJS", "React", "TypeScript", "Docker", "PostgreSQL"]`

Nếu ghép chuỗi, cosine có thể hạ xuống ~0.65–0.7 cho B vì câu dài và nhiều kỹ năng phụ. Với mean pooling:
1. Normalize và sort → chuỗi thứ tự ổn định (`nodejs, react, rest api, ts, docker, postgresql`).
2. Embed từng skill (đều đã chuẩn hoá) → cộng/chi trung bình, sau đó renorm.
3. Cosine A với Job ≈ 1.0 (vì vector trùng), cosine B với Job vẫn ~0.8–0.85 do 3/5 skill trùng, phản ánh tốt hơn mức độ cover.

## Tóm lại
Mean pooling trên từng skill (kèm normalize + dedup + renormalize) tạo vector đại diện ổn định, ít phụ thuộc format câu, giảm nhạy thứ tự/alias và vẫn giữ được độ sắc nét của cosine. Cách này bám sát mức độ cover thực tế giữa danh sách kỹ năng của job và freelancer hơn so với việc ghép tất cả skill thành một chuỗi duy nhất.
