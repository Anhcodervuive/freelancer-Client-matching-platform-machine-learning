🧠 Summary: Match Feature + Machine Learning Pipeline
1️⃣ Mục tiêu hệ thống ML

Hệ thống được thiết kế để dự đoán mức độ phù hợp giữa Job Post và Freelancer, bao gồm:

p_match – xác suất job & freelancer phù hợp

p_freelancer_accept – xác suất freelancer chấp nhận invitation

p_client_accept – xác suất client chọn freelancer

2️⃣ Kiến trúc tổng quan
(1) Embedding Pipeline

Sinh embedding cho:

FULL (title + description)

SKILLS (tính mean embedding các skill)

DOMAIN (category + specialty)

Lưu vào bảng embedding

(2) Match Feature Pipeline

Khi embedding thay đổi hoặc khi chạy CLI seed:

Lấy top-N freelancer/job theo similarity

Tính các feature:

similarity_score

level_gap

timezone_gap_hours

budget_gap (tạm thời ≈ job_budget)

Lưu vào bảng match_feature (upsert)

(3) Machine Learning Training Pipeline

Lấy dataset từ:

match_feature

job_invitation / job_proposal / contract (tùy nhiệm vụ)

Train logistic regression hoặc XGBoost

Xuất model .pkl

(4) ML Prediction Pipeline

Khi tạo match_feature trong tasks.py:

Gọi model để tính p_match / p_freelancer_accept / p_client_accept

Lưu lại vào bảng match_feature

3️⃣ Các feature hiện đang có trong match_feature
Feature Ý nghĩa
similarity_score Mức độ phù hợp embedding FULL/SKILLS/DOMAIN
level_gap Chênh lệch experience job ↔ freelancer
timezone_gap_hours Lệch múi giờ (tạm thời = 0)
budget_gap Tạm thời ≈ ngân sách job (vì chưa có rate freelancer)
p_match Điền bởi ML
p_freelancer_accept Điền bởi ML
p_client_accept Điền bởi ML
last_interaction_at Timestamp hành vi

👉 Usable numeric feature hiện tại: 3–4 → còn ít cho một mô hình mạnh.

4️⃣ Vấn đề phát hiện

Hệ thống hiện mới dùng GAP-based features, nhưng GAP không phản ánh đầy đủ bản chất của job và freelancer.

budget_gap ≈ budget_amount → tính phân biệt thấp.

Thiếu nhiều thông tin quan trọng để ML học được hành vi thật.

5️⃣ Cần bổ sung thêm feature?

→ Có. Rất nên bổ sung.

Một mô hình match chuyên nghiệp (Upwork, LinkedIn, Fiverr) thường dùng 20–80 features.

Hiện bạn mới có 3 feature mạnh, chưa đủ thông tin để ML cho ra chất lượng cao.

6️⃣ Vì sao cần thêm Individual Features (Job-only, Freelancer-only)

GAP mô tả sự khác biệt, nhưng ML cần:

chất lượng hồ sơ freelancer

độ khó job

nhóm ngành

quốc gia

số lượng skill

số job đã hoàn thành

tỷ lệ nhận invite

mức độ cạnh tranh

Những thông tin này KHÔNG thể biểu diễn bằng GAP.

Trong hệ thống recommender thực tế, feature chia làm 2 loại:

Pairwise (Job ↔ Freelancer)

similarity_score

skill_overlap_percentage

timezone_gap

experience_gap

Individual (Job / Freelancer tự thân)

freelancer_skill_count

freelancer_success_rate

freelancer_total_jobs

job_required_skill_count

job_budget

job_category / specialty

Nếu chỉ dùng GAP → mô hình chỉ học được “embedding giống thì match”.

7️⃣ Định hướng cải tiến

Mở rộng match_feature để chứa 10–15 feature mạnh nhất

Viết hàm compute feature đầy đủ trong pipeline match

Xây dataset builder chuẩn cho logistic regression

Train model → gắn vào tasks.py để inference tự động

Sau này nếu có rate hoặc profile nâng cao → cập nhật feature ngay

8️⃣ Kết luận nhanh

Embedding pipeline: đúng và tốt

Match pipeline: đang chạy ổn

match_feature: còn ít feature cho ML

Cần bổ sung thêm nhiều thuộc tính job + freelancer để ML thật sự mạnh

budget_gap giữ lại để tương lai có dữ liệu thật thì dùng

rate_gap đã đúng khi bị loại bỏ

p_match / p_freelancer_accept / p_client_accept chỉ do ML điền

DETAIL:
🔧 3. Đề xuất thêm các trường cho ML

Mình chia thành 3 nhóm:

Job-side features (thuộc tính của job)

Freelancer-side features (thuộc tính của freelancer)

Pairwise features (quan hệ cụ thể giữa job & freelancer)

Ở mỗi dòng mình sẽ ghi:
➡ Targets: p_match / p_freelancer_accept / cả hai

3.1. Job-side features
1️⃣ job_experience_level_num : Int

Map từ enum JobExperienceLevel:

ENTRY → 1

INTERMEDIATE → 2

EXPERT → 3

Targets:

p_match ✅

p_freelancer_accept ✅

Ý nghĩa:

Job càng “hard” (EXPERT) thì chỉ một số freelancer mới dám/đủ sức nhận

Ảnh hưởng đến cả việc freelancer có accept không, và khả năng đôi bên match thành công.

2️⃣ job_required_skill_count : Int

Số lượng skill job yêu cầu
→ lấy từ job_required_skill cho job đó.

Targets:

p_match ✅

p_freelancer_accept ✅

Ý nghĩa:

Job yêu cầu nhiều skill → phức tạp → ít freelancer phù hợp

Freelancer sẽ cân nhắc kỹ hơn để accept.

3️⃣ job_screening_question_count : Int

Số câu hỏi screening trong job_screening_question.

Targets:

p_match ✅

p_freelancer_accept ✅

Ý nghĩa:

Job càng nhiều screening question → thường là job “nghiêm túc” hoặc “khó”

Một số freelancer ngại apply/accept job quá rườm rà.

4️⃣ job_stats_applies : Int

Snapshot từ job_stats.applies tại thời điểm tính feature.

Targets:

p_match ✅

p_freelancer_accept ✅

Ý nghĩa:

Job có nhiều apply → cạnh tranh → dù freelancer accept, khả năng được hire có thể thấp hơn

Một số freelancer tránh những job quá đông ứng viên.

5️⃣ job_stats_offers : Int

Từ job_stats.offers.

Targets:

p_match ✅ (mạnh)

p_freelancer_accept ⚪ (phụ)

Ý nghĩa:

Clients hay gửi offer nhiều → job này mang tính “thật”, không phải spam

Tăng khả năng các match dẫn tới contract (p_match).

6️⃣ job_stats_accepts : Int

Từ job_stats.accepts (số offer đã được accept).

Targets:

p_match ✅ (mạnh)

p_freelancer_accept ⚪ (phụ)

Ý nghĩa:

Job/client có “lịch sử hire thành công” → conversion tốt → tăng xác suất match.

3.2. Freelancer-side features
7️⃣ freelancer_skill_count : Int

Số lượng skill trong freelancer_skill_selection (is_deleted = 0).

Targets:

p_match ✅

p_freelancer_accept ✅

Ý nghĩa:

Freelancer nhiều skill → đa năng, có thể phù hợp nhiều job

Cũng có thể là tín hiệu “senior“.

8️⃣ freelancer_stats_applies : Int

Từ freelancer_stats.applies.

Targets:

p_match ✅

p_freelancer_accept ✅

Ý nghĩa:

Freelancer hay apply → chủ động, hunting job

Có pattern: freelancer apply nhiều nhưng ít được hire (kết hợp với accepts).

9️⃣ freelancer_stats_offers : Int

Từ freelancer_stats.offers.

Targets:

p_match ✅ (mạnh)

p_freelancer_accept ⚪

Ý nghĩa:

Freelancer hay được gửi offer → profile attractive

Dễ dẫn đến match thành công.

🔟 freelancer_stats_accepts : Int

Từ freelancer_stats.accepts.

Targets:

p_match ✅

p_freelancer_accept ✅

Ý nghĩa:

Freelancer hay accept offer (đã có trước đây) → “easy going”, ít kén

Higher base-rate accept job.

1️⃣1️⃣ freelancer_invite_accept_rate : Float

Tính từ lịch sử:

invitation_accepted / max(1, invitations_sent)

Targets:

p_freelancer_accept ✅✅ (rất mạnh)

p_match ✅

Ý nghĩa:

Đây là feature trực tiếp nhất cho mô hình p_freelancer_accept:

Người từng accept 80% invite → khả năng accept tiếp theo rất cao

Người chỉ accept 5% → thường từ chối.

1️⃣2️⃣ freelancer_country_region : String / Enum

Region (VD: VN, SEA, EU, US…) rút gọn từ country (tránh high-cardinality).

Targets:

p_match ✅

p_freelancer_accept ⚪

Ý nghĩa:

Một số job ưu tiên freelancer trong khu vực/múi giờ cụ thể

Region kết hợp với timezone_gap_hours là tín hiệu tốt.

3.3. Pairwise features (job–freelancer cụ thể)
1️⃣3️⃣ skill_overlap_count : Int

Số skill trùng giữa:

job_required_skill

freelancer_skill_selection

Targets:

p_match ✅✅

p_freelancer_accept ✅

Ý nghĩa:

Nhiều skill trùng → freelancer “feel” job phù hợp → dễ apply/accept

Cũng tăng khả năng client chọn vì matching tốt.

1️⃣4️⃣ skill_overlap_ratio : Float

skill_overlap_count / max(1, job_required_skill_count)

Targets:

p_match ✅✅

p_freelancer_accept ✅

Ý nghĩa:

Tỷ lệ cover skill yêu cầu:

1.0 → cover 100% skill

0.5 → cover 50% skill

Mạnh hơn count đơn thuần khi job size khác nhau.

1️⃣5️⃣ has_past_collaboration : Bool

true nếu:

đã có contract giữa client của job và freelancer này trước đó.

Targets:

p_match ✅✅

p_freelancer_accept ✅

Ý nghĩa:

Hai bên từng làm việc chung → cực kỳ tăng xác suất:

Client hire lại (p_match)

Freelancer accept job (đã biết client này ok).

1️⃣6️⃣ past_collaboration_count : Int

Số contract đã hoàn thành giữa 2 bên.

Targets:

p_match ✅✅

p_freelancer_accept ✅

Ý nghĩa:

Một job mới với client cũ: collaboration count cao → gần như “auto match”.

1️⃣7️⃣ has_viewed_job : Bool

1 nếu freelancer đã từng view job này (log từ match_interaction type = JOB_VIEW).

Targets:

p_freelancer_accept ✅

p_match ⚪

Ý nghĩa:

Freelancer đã nhìn thấy job → bước đầu quan tâm

Từ đó đến accept là một bước nữa.

🎯 4. Trường nào dùng cho mô hình nào?

Tóm nhanh:

Cho p_match (job & freelancer cuối cùng có contract / hợp tác thành công không):

similarity_score

level_gap

timezone_gap_hours

budget_gap (về sau có rate)

job_experience_level_num

job_required_skill_count

job_stats_applies / offers / accepts

freelancer_skill_count

freelancer_stats_applies / offers / accepts

freelancer_invite_accept_rate

freelancer_country_region

skill_overlap_count / skill_overlap_ratio

has_past_collaboration / past_collaboration_count

last_interaction_at (suy ra “freshness”)

👉 Gần như tất cả feature đều hữu ích cho p_match.

Cho p_freelancer_accept (freelancer có accept invitation này không):

similarity_score

level_gap

timezone_gap_hours

job_experience_level_num

job_required_skill_count

job_screening_question_count

job_stats_applies (job đông ứng viên có thể làm freelancer lười apply)

freelancer_skill_count

freelancer_stats_applies / accepts

freelancer_invite_accept_rate (feature chủ lực)

skill_overlap_count / ratio

has_past_collaboration / past_collaboration_count

has_viewed_job

last_interaction_at (ví dụ đã tương tác gần đây)

👉 Đặc biệt quan trọng:
freelancer_invite_accept_rate, skill_overlap_ratio, has_past_collaboration, similarity_score.

🏷️ 5. Lấy nhãn (label) như thế nào?

Mình chia rõ cho từng mô hình.

5.1. Nhãn cho mô hình p_freelancer_accept

Mục tiêu:

Dự đoán: nếu gửi invitation {job, freelancer} thì freelancer có ACCEPT hay không?

a) Nguồn nhãn

Lấy từ bảng job_invitation (và có thể kết hợp với proposal/contract).

Positive (label = 1):

JobInvitation.status == ACCEPTED

Hoặc invitation đó dẫn đến:

Proposal được tạo → Offer → Contract

(tùy bạn có muốn “coi như accept” khi vào contract luôn không)

Negative (label = 0):

JobInvitation.status IN (DECLINED, EXPIRED)

Hoặc INVITATION_SENT nhưng sau N ngày không trả lời (coi như ignore → negative).

Bỏ qua:

INVITATION_SENT nhưng vẫn đang trong window phản hồi (chưa đủ thời gian)

Các record test hoặc spam.

b) Build dataset

Mỗi dòng dataset = 1 invitation:

(job_id, freelancer_id)
→ join sang match_feature để lấy toàn bộ feature tại thời điểm đó
→ label_accept = 0 hoặc 1

c) Train

X = các feature trong match*feature (trừ p_match, p*\*\_accept)

y = label_accept

Mô hình: logistic regression / XGBoost

Kết quả:

predict p_freelancer_accept cho mọi cặp (job, freelancer) mà bạn xét.

5.2. Nhãn cho mô hình p_match

Mục tiêu:

Dự đoán: cặp job–freelancer này cuối cùng có “match thành công” (hire/thực sự làm việc) hay không?

a) Định nghĩa “match thành công”

Bạn có thể chọn 1 trong 2:

Mức strong:

Có contract với status IN (ACTIVE, COMPLETED, CANCELLED_AUTO_RELEASED v.v.)

Mức medium:

Hoặc: JobProposal.status == HIRED

Hoặc: JobOffer.status == ACCEPTED

Tùy bạn định nghĩa, nhưng nên thống nhất 1 tiêu chí rõ.

b) Nguồn nhãn

Positive (label = 1):

Các cặp (job_id, freelancer_id) thỏa điều kiện “match thành công” ở trên.

Negative (label = 0):

Các cặp đã từng:

Có proposal SUBMITTED nhưng bị DECLINED / không được hire

Có offer SENT nhưng DECLINED / EXPIRED

Có invitation ACCEPTED nhưng không dẫn tới contract sau N ngày.

Bỏ qua:

Cặp còn đang “pending” (proposal SUBMITTED, offer SENT nhưng chưa rõ outcome).

c) Build dataset

Từ logs:

lấy tất cả cặp (job_id, freelancer_id) có hoạt động (proposal, offer, contract).

Gắn label:

positive / negative theo rule trên.

Join sang match_feature:

để lấy snapshot feature (lưu ý thời gian — nếu bạn muốn rất chuẩn, sẽ cần snapshot theo thời điểm, nhưng giai đoạn đầu có thể dùng gần-thời-điểm).

5.3. Lưu ý quan trọng

Không dùng các cột p_match, p_freelancer_accept, p_client_accept làm label.
→ Chúng chỉ là nơi ghi lại output model.

Label luôn lấy từ:

trạng thái cuối cùng của Invitation / Proposal / Offer / Contract.
