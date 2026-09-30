# NAMING AUDIT — Tên phương pháp đề xuất (DENTEX Selective Classification)

**Ngày:** 2026-09-30
**Trạng thái protocol:** v6, §5.6 = **HOÃN naming**, do ablation §9.6 quyết định
**Phạm vi audit:** chỉ đánh giá tên. **Không** đề xuất thay đổi công thức, tham số, metric, baseline.
**Không phải DEVIATION:** audit này không chọn tên và không sửa §5.6. Việc *chọn* tên trước khi §9.6 chạy mới là vi phạm.

---

## 0. Công thức cuối — tập thành phần bắt buộc phải khớp với acronym

Theo §5.4:

```
Φ_S(S(x)) = ECDF của S trên validation fold của vòng đó
Φ_M(g(x)) = ECDF của g trên validation fold của vòng đó
g(x)      = − min_k (z − μ_k)ᵀ Σ_shrunk⁻¹ (z − μ_k),   z ∈ ℝ^64 (PCA, d khóa = 64)
R_α(x)    = α · Φ_S(S(x)) + (1 − α) · Φ_M(g(x)),        α ∈ [0,1]
Quyết định: R_α(x) ≥ τ → ACCEPT ; R_α(x) < τ → ABSTAIN
```

Liệt kê đầy đủ thành phần — acronym chỉ được khai triển thành đúng tập này (nguyên tắc §5.6):

| # | Thành phần | Bản chất |
|---|---|---|
| C1 | `S(x)` = max softmax probability | tín hiệu **confidence** từ logit head |
| C2 | `g(x)` = −min_k Mahalanobis² | tín hiệu **typicality trong feature space**, class-conditional Gaussian trên PCA-64 |
| C3 | `Φ` = ECDF per-fold, average rank cho ties | phép **rank transform**, áp đồng nhất cho cả hai tín hiệu |
| C4 | `α` tổ hợp lồi, fit per-fold trên validation fold | **convex combination**, 1 tham số, quét lưới bước 0,05 |
| C5 | `τ` ngưỡng accept/abstain | **gate** (selective classification) |

**Không có** Earth Mover's Distance. **Không có** Energy. **Không có** optimal transport. **Không có** manifold learning. **Không có** tầng chiếu học được (§5: PCA chứ không phải learned projection).

---

## 1. Yêu cầu 1 — `EMD` có hợp lệ không?

**Kết luận: KHÔNG. Loại bỏ hoàn toàn.** Tên này sai dưới **cả hai** cách khai triển khả dĩ:

| Khai triển | Trạng thái trong công thức cuối | Phán quyết |
|---|---|---|
| `E` = **Energy** (ý nghĩa gốc, theo §5.6) | Energy đã bị bỏ khỏi `R(x)`; hiện chỉ còn là **baseline** ở §6.1 và **ứng viên thành phần thứ 3** ở §9.6 | ❌ Tên chứa một thành phần không có trong công thức |
| `E` = **Earth Mover's Distance** (suy diễn sai, 2026-09-30) | Không hề xuất hiện. EMD là khoảng cách **phân phối–phân phối**; Mahalanobis là **điểm–phân phối** | ❌ Sai loại toán học |

**Rủi ro cộng thêm, đáng chú ý:** `Φ` là ECDF. Một reviewer quen optimal transport thấy "EMD" + "CDF-based transform" rất dễ suy ra rằng `Φ` là một cấu trúc Wasserstein (EMD 1-D *thật sự* tính được bằng tích phân hiệu hai CDF). Nó không phải — `Φ` chỉ là rank transform. Tên này không chỉ trống nghĩa mà còn **chủ động gợi sai** về bản chất C3.

`D` (Discrepancy / Distance) cũng nên bỏ cùng: "discrepancy" trong văn liệu domain adaptation (MMD, CORAL) chỉ khoảng cách giữa **hai phân phối**, lại trùng đúng cái nhầm lẫn trên.

> **Hệ quả cho tiêu đề cũ:** cụm *"via Latent Manifold Discrepancy"* mắc cùng lỗi. Mahalanobis trên PCA-64 với một Gaussian mỗi lớp là **mô hình elliptical đơn**, không phải mô hình manifold. Tiêu đề chính thức ở §1 plan (*"Selective Classification for Dental Panoramic Radiographs…"*) không hứa gì sai — giữ nguyên.

---

## 2. Yêu cầu 2 — `PMR-Gate` (Percentile-Mapped Reliability Gate) có overclaim không?

**Kết luận: CÓ, overclaim ở hai chỗ độc lập. Loại.**

### 2.1 `Reliability` — vi phạm §9 (danh sách "bài báo KHÔNG tuyên bố")

"Reliability" trong văn liệu calibration có nghĩa kỹ thuật chặt: reliability diagram, ECE — mức khớp giữa confidence và accuracy thực tế. `R_α(x)` **không phải xác suất đã hiệu chỉnh**: nó là điểm **thứ hạng** trong khoảng [0,1] do ECDF sinh ra, giá trị 0,8 không có nghĩa "đúng 80%".

Primary endpoint của bài là `AURC_CV` — metric **xếp hạng**, không phải metric calibration. Bài không đo ECE làm primary. Đặt tên "Reliability" là hứa một tính chất mà thiết kế không thiết lập và không đo, đụng trực tiếp điều khoản §9: *không tuyên bố cơ chế ước lượng bất định mới về toán học*.

### 2.2 `Percentile-Mapped` đặt contribution vào sai chỗ — vi phạm §4

Đây là lỗi nghiêm trọng hơn. Đặt `Φ` làm định ngữ đầu tên nghĩa là tuyên bố rank transform là đóng góp. Nhưng chính protocol đã chứng minh ngược lại:

- **§5.5:** tại α=1, `R_1 = Φ_S(S)` và `AURC_f(R_1) = AURC_f(MSP)` **bằng chính xác** trong từng fold. Tức `Φ` **không có hiệu ứng nào** trong một fold — nó là phép biến đổi đơn điệu.
- **§4:** *"ECDF normalization does not change the within-fold ranking or within-fold AURC"*, và **CẤM mô tả nó như cải tiến của baseline** — nó là **aggregation device**.

Đặt tên theo `Φ` tạo ra một câu hỏi phản biện tự sát: *"Sanity check của chính các tác giả chứng minh thành phần được đặt tên là no-op trong fold — vậy tên chỉ cái gì?"*

Vai trò thật của `Φ`: làm hai tín hiệu **khác thang đo** (`S ∈ [0,1]`, `g ∈ ℝ⁻` không chặn) trở nên **tổ hợp được** bằng một tham số `α` duy nhất. Đó là điều kiện kỹ thuật cần thiết, không phải nguồn hiệu năng.

### 2.3 Không khớp nguyên tắc §5.6

`PMR` không khai triển ra C1 (softmax confidence) hay C2 (Mahalanobis) — hai tín hiệu **thực sự** tạo nên điểm. Vi phạm trực tiếp *"acronym khai triển thành đúng tập thành phần trong công thức cuối"*.

---

## 3. Yêu cầu 3 — Ứng viên dựa trên phương trình, kèm xếp hạng rủi ro

### 3.1 Bảng ứng viên

`Cov` = thành phần được acronym bao phủ. `Sót` = thành phần có trong công thức nhưng vắng trong tên (chấp nhận được nếu là C3/C5). `Thừa` = thành phần tên hứa nhưng công thức không có (**không chấp nhận được**).

| # | Tên | Khai triển | Cov | Thừa | Ghi chú |
|---|---|---|---|---|---|
| N1 | **CM-Gate** | Confidence–Mahalanobis Gate | C1,C2,C5 | — | Tối giản. Nói đúng hai tín hiệu + cơ chế. Không hứa gì thêm |
| N2 | **CMC-Gate** | Confidence–Mahalanobis Convex Gate | C1,C2,C4,C5 | — | Thêm "Convex" — đúng theo §5.4, và ngụ ý đơn điệu (`∂R/∂Φ_S = α > 0`) |
| N3 | **RCM-Gate** | Rank-Combined Confidence–Mahalanobis Gate | C1,C2,C3,C4,C5 | — | Bao phủ đủ nhất. "Rank-Combined" trung tính hơn "Percentile-Mapped" vì đặt `Φ` ở vị trí bổ ngữ cho *combination*, không phải head noun |
| N4 | **SMC** | Softmax–Mahalanobis Convex score | C1,C2,C4 | — | Bỏ "Gate" → phù hợp nếu muốn nhấn *điểm* thay vì *quyết định* |
| N5 | **α-CM** | alpha-weighted Confidence–Mahalanobis | C1,C2,C4 | — | Nhấn `α` là tham số duy nhất. Hơi kỹ thuật, khó đọc thành tiếng |
| N6 | **CM-Abstain** | Confidence–Mahalanobis Abstention rule | C1,C2,C5 | — | "Abstain" là từ vựng chuẩn của selective classification, chính xác hơn "Gate" |
| N7 | **DSC-Gate** | Dual-Signal Confidence Gate | C1?,C4,C5 | — | "Dual-Signal" đúng nhưng không nói *tín hiệu nào* → mờ, không sai |
| N8 | **ECM-Gate** | Energy–Confidence–Mahalanobis Gate | C1,C2,C5 + Energy | ⚠️ Energy | **Chỉ dùng được nếu §9.6 kết luận nhận Φ_E vào công thức.** Hiện tại = thừa |

### 3.2 Xếp hạng rủi ro gây hiểu nhầm (cao → thấp)

| Hạng | Tên | Rủi ro | Cơ chế hiểu nhầm |
|---|---|---|---|
| 🔴 **1** | **EMD-Gate** | **Loại bỏ** | Thành phần thừa dưới mọi cách đọc; còn gợi sai rằng `Φ` là cấu trúc optimal transport |
| 🔴 **2** | *bất kỳ tên chứa* **Reliability / Calibrated / Trustworthy / Safe** | **Loại bỏ** | Hứa tính chất calibration mà bài không đo làm primary; đụng §9 |
| 🔴 **3** | *bất kỳ tên chứa* **Manifold** | **Loại bỏ** | Mahalanobis + một Gaussian/lớp trên PCA-64 không phải mô hình manifold |
| 🔴 **4** | *bất kỳ tên chứa* **OOD / Novelty / Anomaly** | **Loại bỏ** | §1 bảng từ vựng cấm "OOD"; §9 cấm tuyên bố phát hiện OOD |
| 🟠 **5** | **PMR-Gate** | Cao | (a) "Reliability" overclaim; (b) đặt tên theo thành phần mà §5.5 chứng minh là no-op trong fold; (c) không nêu C1/C2 |
| 🟠 **6** | **N8 ECM-Gate** | Cao *có điều kiện* | Đúng ⟺ §9.6 nhận Energy. Dùng trước khi §9.6 chạy = lặp lại đúng lỗi của EMD |
| 🟡 **7** | **N7 DSC-Gate** | Trung bình | "Dual-Signal" mơ hồ; "Confidence Gate" có thể bị đọc thành calibrated confidence |
| 🟡 **8** | **N3 RCM-Gate** | Trung bình–thấp | "Rank-Combined" vẫn kéo chú ý về `Φ`; nhẹ hơn PMR vì không phải head noun. Cần một câu trong bài nói rõ `Φ` là aggregation device (§4) |
| 🟡 **9** | **N5 α-CM** | Thấp–trung bình | Không gây hiểu nhầm về toán, nhưng `α` không được giải thích trong tiêu đề → khó đọc độc lập |
| 🟢 **10** | **N4 SMC** | Thấp | "Softmax" cụ thể hơn "Confidence", tránh hẳn liên tưởng calibration |
| 🟢 **11** | **N2 CMC-Gate** | Thấp | Mọi chữ khai triển ra một thành phần thật. "Convex" kiểm chứng được từ §5.4 |
| 🟢 **12** | **N6 CM-Abstain** | Thấp | Dùng từ vựng chuẩn của selective classification |
| 🟢 **13** | **N1 CM-Gate** | **Thấp nhất** | Không có chữ nào hứa quá. Nhược điểm duy nhất: nhàm, và không phân biệt với "Mahalanobis + MSP" nói chung — nhưng đó chính là **điều bài đang làm** |

### 3.3 Quan sát then chốt về cấu trúc quyết định

Naming **không thể** quyết định bây giờ, vì §9.6 còn để mở việc `Φ_E` (Energy) có vào công thức hay không:

- §9.6 **không nhận** Energy → name hợp lệ nằm trong {N1, N2, N3, N4, N5, N6}
- §9.6 **nhận** Energy → công thức thành `R = α·Φ_S + β·Φ_M + γ·Φ_E`, khi đó **mọi** tên ở nhóm trên trở thành thiếu thành phần, và N8 (`ECM-Gate`) mới hợp lệ

Tức thứ tự đúng là: **§9.6 → khóa tập thành phần → chọn tên**. Không có đường ngược.

⚠️ Lưu ý §9.6 hiện khai báo `R = α·Φ_S + β·Φ_M + γ·Φ_E` với **ba** trọng số. Nếu ablation này được nhận vào công thức chính thì số tham số hậu nghiệm tăng từ 1 lên 2 (`α`, và một trong `β`/`γ` sau ràng buộc tổng = 1) — đó là **thay đổi tập tham số**, cần DEVIATION LOG theo §2 (*"Không thêm tham số nào ngoài hai bảng này"*), không chỉ là đổi tên.

---

## 4. Kết luận theo đúng 3 yêu cầu

1. **`EMD` — loại bỏ.** Sai dưới cả hai cách khai triển; còn gợi sai về bản chất của `Φ`.
2. **`PMR-Gate` — loại bỏ.** "Reliability" overclaim (§9); "Percentile-Mapped" đặt contribution vào thành phần mà §5.5 chứng minh là no-op trong fold (§4); không nêu hai tín hiệu thật.
3. **Shortlist theo rủi ro thấp nhất:** `CM-Gate` (N1) < `CM-Abstain` (N6) ≈ `CMC-Gate` (N2) < `SMC` (N4). Rủi ro cao nhất: `EMD-Gate`, rồi mọi tên chứa *Reliability / Manifold / OOD*, rồi `PMR-Gate`.

**Trạng thái §5.6 giữ nguyên: HOÃN.** Audit này chỉ thu hẹp không gian lựa chọn, không chọn.

---

## 5. PHẦN VƯỢT PHẠM VI — khai báo theo §0.5

Ba phần dưới đây **không** nằm trong 3 yêu cầu. Nêu ra để anh quyết định giữ hay bỏ; em sẵn sàng xóa khỏi file.

**(a) Cân nhắc: có cần acronym không?**
Một acronym có thương hiệu ngầm phát tín hiệu *"đây là mechanism mới"*. §9 lại quy định bài **không** tuyên bố cơ chế ước lượng bất định mới về toán học. Đóng góp thật của bài — theo đúng những gì §7/§4 cho phép phát biểu — là một **đánh giá selective classification có kiểm soát rò rỉ trên ảnh nha khoa**, cộng một điểm tổ hợp đơn giản một tham số. Phương án rủi ro thấp nhất có thể là **không đặt acronym**, chỉ gọi mô tả: *"the convex rank-combined score"* / *"the combined confidence–Mahalanobis score"*. Cách này triệt tiêu toàn bộ rủi ro ở Mục 3.2 và khớp với văn phong khiêm tốn mà §1 và §9 đang áp cho toàn bài.

**(b)** Tiêu đề cũ chứa *"Latent Manifold Discrepancy"* mắc cùng lỗi "Manifold" — xem ghi chú ở Mục 1.

**(c)** Rủi ro tham số của §9.6 — xem cảnh báo ở Mục 3.3.
