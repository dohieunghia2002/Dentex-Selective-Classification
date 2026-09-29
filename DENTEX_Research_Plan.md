# RESEARCH PLAN v6 — LOCKED PROTOCOL
## Selective Classification for Dental Panoramic Radiographs in Differential Diagnosis: Mitigating Overconfident Failures Under Anatomical and Metallic Artifacts

**Phiên bản:** v6 — bản khóa cuối, thay thế v5
**Ngày khóa:** 2026-09-29
**Dataset:** DENTEX (MICCAI 2023), **chỉ** subset `quadrant_enumeration_disease` của tập *training* công khai
**Mục tiêu xuất bản:** CMPB / IEEE JBHI / IEEE ISBI / Diagnostics / MICCAI UNSURE

**Thay đổi so với v5:** chỉ **quy tắc tổng hợp (aggregation protocol)** và cách gọi tên. Không đổi câu hỏi nghiên cứu, dataset, kiến trúc, số lần huấn luyện hay bất kỳ thí nghiệm nào. Xem §15.

---

> ## ⚠️ QUY TẮC VẬN HÀNH
>
> 1. Tài liệu này là **pre-registration**. Mọi tham số, metric, baseline, ablation đã khóa. Không sửa sau khi nhìn kết quả.
> 2. Mọi sai lệch ghi vào **§14 DEVIATION LOG**, kèm ngày và lý do, **trước khi** áp dụng.
> 3. **Tham số huấn luyện đã KHÓA HOÀN TOÀN (§5.3-A).** Chỉ **tham số hậu nghiệm (§5.3-B)** được chọn trên validation fold.
> 4. **Quy tắc tổng hợp đã KHÓA (§7).** Lựa chọn giữa primary và secondary **không được phụ thuộc vào kết quả quan sát được**.
> 5. Những gì bài báo **KHÔNG** tuyên bố: §12.3.
> 6. **Mọi phát biểu so sánh với văn liệu ("chưa ai làm", "đầu tiên") bị CẤM**, trừ khi có systematic search được ghi lại trong phụ lục.

---

# §0. TỪ VỰNG CHUẨN

| Thuật ngữ | Nghĩa chính xác |
|---|---|
| **Train folds** | 3 fold dùng huấn luyện trong một vòng CV cụ thể |
| **Validation fold** | 1 fold dùng hiệu chỉnh tham số hậu nghiệm trong vòng đó |
| **Test fold** | 1 fold được giữ lại trong vòng đó |
| **OOF prediction set** | Tập hợp 3.523 dự đoán — mỗi patch mang **đúng một** dự đoán, sinh từ vòng CV mà ảnh chứa nó nằm ở test fold |
| **Clinical annotation cohort** | Tập patch được gán nhãn cạm bẫy. Dưới phương án pilot (a) của §11.2 nó **bằng** OOF prediction set (patch pilot đến từ ngoài 705 ảnh); dưới phương án (b) nó là OOF prediction set **trừ** 15 patch pilot |

> ### ⚠️ Phân biệt bắt buộc: tập dữ liệu ≠ đơn vị đánh giá
>
> **OOF prediction set là một TẬP HỢP DỰ ĐOÁN, không phải đơn vị tính metric.**
>
> Mỗi fold được chấm điểm bởi **một mô hình khác nhau**, nên điểm số từ các fold khác nhau **không được giả định là khả so sánh trực tiếp** (§7.0). Vì vậy:
>
> - **Primary endpoint** tính **trong từng fold** rồi macro-average (§7.1–7.4).
> - **Secondary/sensitivity** ghép toàn bộ, nhưng **chỉ sau khi chuẩn hóa hạng theo fold áp đồng nhất cho mọi phương pháp** (§7.6).
>
> Không có metric nào trong bài được tính bằng cách sort điểm **thô** của cả 5 fold chung một bảng.

---

# §1. CÂU HỎI NGHIÊN CỨU

## 1.1 Câu hỏi chính

> Trong bối cảnh **chẩn đoán phân biệt 4 thể bệnh lý trên các răng mang nhãn chẩn đoán X-quang của chuyên gia**, một cơ chế gating kết hợp độ tự tin softmax với độ tương thích manifold có làm giảm **sai sót tự tin thái quá** so với các phương pháp selective prediction hiện có hay không — đặc biệt trên răng có **cạm bẫy giải phẫu và nhiễu kim loại**?

## 1.2 Thuật ngữ bắt buộc dùng đúng

| ❌ KHÔNG dùng | ✅ Dùng | Lý do |
|---|---|---|
| "confirmed disease" | **"teeth carrying an expert radiographic diagnosis annotation"** | Nhãn DENTEX là chẩn đoán trên phim, **không có xác nhận mô bệnh học** |
| "single-center" | **"a single publicly released dataset"** | §1.4 |
| "ground truth" (không điều kiện) | "reference annotation" | Cùng lý do |
| "healthy teeth" | "unannotated teeth" | §12.1 |
| "healthy" (lớp thứ 5) | **"no DENTEX-category finding observed"** | §12.1 |
| "OOD" | "near-distribution" / "unlabeled" | §12.1 |
| "test set" (cho kết quả cuối) | **"OOF prediction set"**, kèm quy tắc tổng hợp | §0 |
| **"bias" (cho hiện tượng ghép fold)** | **"cross-fold scale confounding" / "aggregation artifact"** | §7.0 — chưa chứng minh được estimator bias theo nghĩa thống kê |

## 1.3 Nằm NGOÀI phạm vi

| Ngoài phạm vi | Lý do |
|---|---|
| Phát hiện false positive trên răng lành | DENTEX không có nhãn "healthy"; gán nhãn suy diễn là Negative Sampling Fallacy |
| Object detection / phát hiện răng | Thiết kế dùng ground-truth box, phân loại patch |
| OOD detection | Răng không nhãn **không** phải OOD; chúng có thể mang bệnh |
| External validation trên dataset khác | Khai báo ở Limitations |

## 1.4 Mô tả dataset — official split vs internal CV

| | Mô tả |
|---|---|
| **Official DENTEX split** | Challenge có training / validation / test riêng. Nhãn validation và test **không công khai đầy đủ** |
| **Cái chúng tôi dùng** | **Chỉ** 705 ảnh của subset `quadrant_enumeration_disease` thuộc tập **training** công khai |
| **Internal 5-fold CV** | Toàn bộ chia fold diễn ra **bên trong** 705 ảnh đó |

DENTEX là **một dataset công khai duy nhất**, nhưng dữ liệu bên trong **không nhất thiết đến từ một cơ sở duy nhất**.

- ✅ Viết: *"a single publicly released dataset"*
- ❌ Không viết: *"single-center"*
- ⚠️ Nếu bản thảo nêu số cơ sở đóng góp hoặc model máy chụp, phát biểu đó **bắt buộc trích dẫn tài liệu gốc DENTEX** và đối chiếu nguồn sơ cấp (§16-D).

> **PHẢI GHI TRONG BÀI:** *"Chúng tôi không sử dụng official DENTEX challenge split; kết quả trong bài **không so sánh được** với bảng xếp hạng của challenge."*

---

# §2. DỮ LIỆU

## 2.1 Phân bố nhãn

| ID | Lớp | Số annotation | Số patch | Tỷ lệ (annotation) |
|---|---|---|---|---|
| 0 | `Impacted` | 604 | 604 | 17,1% |
| 1 | `Caries` | 2.189 | 2.186 | 62,0% |
| 2 | `Periapical_Lesion` | **158** | **157** | **4,5%** ⚠️ |
| 3 | `Deep_Caries` | 578 | 576 | 16,4% |
| | **Tổng** | 3.529 | 3.523 | |

> Cột "Số patch" bổ sung ngày 2026-09-29 (xem §14): trước đây bảng chỉ ghi số cấp annotation. Số patch là số sau khi loại 3 vị trí hộp mang >1 annotation.

- 3.529 annotation trên **3.526 vị trí hộp**, trên **705 ảnh panorama**.
- **3 vị trí hộp mang >1 annotation (3 box positions with >1 annotation; 0,09%) → loại bỏ → 3.523 patch.** (Trước đây ghi "3 hộp mang 2 nhãn" — xem §14.) `image_id` 306 và 536 thực sự mang hai nhãn khác nhau (Periapical Lesion + Deep Caries; Caries + Deep Caries); `image_id` 447 là cùng một nhãn Caries lặp lại.
- **705 ảnh, trong đó 678 ảnh có ít nhất một annotation**; 27 ảnh không có annotation nào (xem §7.3, §14).
- Mật độ **5,20 patch trên mỗi ảnh có nhãn** (5,00 nếu tính trên cả 705).

## 2.2 Cỡ mẫu theo fold — con số phải nhớ khi đọc kết quả

```
Mỗi test fold  ≈ 141 ảnh  ≈ 705 patch
               ≈ 136 ảnh có nhãn  (= cỡ mẫu của metric cấp ảnh, §7.3)
Periapical mỗi test fold ≈ 31 mẫu   (157/5; trước đây ghi ≈32 từ 158/5 — xem §14, 2026-09-29, mục §2.1, §7.6)
```

> Con số ≈31 chỉ là trung bình. Iterative stratification (§3.1) cân bằng số **ảnh chứa** Periapical (≈23–24 ảnh mỗi fold), **không** cân bằng số **patch**; một ảnh có thể mang nhiều răng Periapical, nên số patch Periapical mỗi fold trải rộng hơn con số trung bình gợi ý. Phân bố thật được báo cáo nguyên trạng ở §10.1 sau khi sinh `folds.json`. Đây là thêm một lý do để phân tích lớp hiếm nằm ở nhánh secondary §7.6 (như §7.4 đã nêu).

Đây là cỡ mẫu của **primary endpoint trong từng fold**. Phân tích lớp hiếm và nhóm con ở mức per-fold sẽ có CI rộng — xem §7.4, §11.1.

## 2.3 Shortcut đã biết của `Impacted`

Răng ngầm nhận diện được từ vị trí và hình thái crop → accuracy dự kiến cao. Không phải lỗi cần sửa; xử lý ở §7.4.

---

# §3. CHIA DỮ LIỆU & KIỂM SOÁT RÒ RỈ

## 3.1 Chia fold — tất định, multilabel stratification

**Đơn vị chia: ảnh panorama (`image_id`).**

Mỗi ảnh biểu diễn bằng **vector nhãn nhị phân 4 chiều** — lớp nào *có xuất hiện* trong ảnh đó. Dùng **Iterative Stratification cho dữ liệu đa nhãn** (Sechidis et al., 2011), **không** phân tầng theo lớp chiếm đa số.

```
Thuật toán:  iterstrat.MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
             (package iterative-stratification, §3.5; Sechidis bậc một)
Đầu vào:     ma trận nhãn ảnh-lớp (705 × 4), nhị phân
Seed:        cố định, ghi trong folds.json
Đầu ra:      folds.json  →  COMMIT VÀO REPO, KHÔNG SINH LẠI
```

> Trước đây ghi `IterativeStratification(n_splits=5, order=1)` (cú pháp skmultilearn) — xem §14, 2026-09-29.

> **KHÔNG có quy tắc "fold nào có <20 Periapical thì chia lại".** Đó là bậc tự do phụ thuộc dữ liệu. Phân bố thực tế được **báo cáo nguyên trạng**.

## 3.2 Cấu hình 5-fold xoay vòng

| Vòng CV | Train folds | Validation fold | Test fold |
|---|---|---|---|
| 1 | F2, F3, F4 | F5 | F1 |
| 2 | F3, F4, F5 | F1 | F2 |
| 3 | F4, F5, F1 | F2 | F3 |
| 4 | F5, F1, F2 | F3 | F4 |
| 5 | F1, F2, F3 | F4 | F5 |

**Lý do dùng 5-fold — ba điểm, không điểm nào giả định khả so sánh liên model:**

1. **Mọi ảnh đóng góp đúng một lần vào đánh giá.** Không ảnh nào bị bỏ phí, và ước lượng không bị neo vào một phép chia 15% tùy ý.
2. **`AURC_CV` trung bình trên 5 lần huấn luyện** nên ít nhạy với một phép chia không may hơn so với single split.
3. **Nhánh secondary (§7.6)** giữ được cỡ mẫu đầy đủ cho lớp hiếm và nhóm con — nơi per-fold không đủ.

> Lưu ý: 5-fold **không phải** để cứu power cho so sánh chính. Thiết kế ghép cặp (§8) đã lo phần đó.

## 3.3 BẢNG KIỂM SOÁT RÒ RỈ

| Thành phần | Fit trên | TUYỆT ĐỐI KHÔNG dùng |
|---|---|---|
| Trọng số backbone | 3 train folds | validation fold, test fold |
| PCA projection | 3 train folds | validation fold, test fold |
| μ_k (class centroids) | 3 train folds | validation fold, test fold |
| Σ_shrunk (tied, Ledoit-Wolf) | 3 train folds | validation fold, test fold |
| **μ₀, Σ₀ của RMD (background Gaussian)** | **3 train folds** | validation fold, test fold |
| Không gian con chính của ViM | 3 train folds | validation fold, test fold |
| Φ_S, Φ_M (ECDF) | validation fold | test fold |
| **Φ_method của nhánh secondary (§7.6)** | **validation fold** | test fold |
| α (trọng số tổ hợp) | validation fold | test fold |
| T (temperature scaling) | validation fold | test fold |
| **τ (ngưỡng vận hành)** | **validation fold** | test fold — §3.4 |
| **`d` (chiều PCA)** | **KHÓA TRƯỚC — không fit ở đâu cả** (§5.3-A) | — |
| LR, augmentation, epochs | **KHÓA TRƯỚC — không fit ở đâu cả** | — |

## 3.4 Ngưỡng τ — hai con số, hai vai trò

| Ký hiệu | Cách tính | Vai trò |
|---|---|---|
| `OracleCov-Risk@{70,80,90}` | Sắp xếp **trong từng test fold**, cắt tại coverage mục tiêu, rồi **macro-average 5 fold** | **CHỈ LÀ DIAGNOSTIC.** Ghi nhãn *"oracle coverage, not achievable in deployment"* |
| `ValCalibrated-τ{70,80,90}` | τ chọn **trên validation fold** để đạt coverage mục tiêu trên val → **freeze** → áp lên test fold của cùng vòng → **macro-average 5 fold** | **KẾT QUẢ CHÍNH** |

**Quy định báo cáo cho `ValCalibrated-τ` (bắt buộc đủ ba):**
1. Macro-average không trọng số của **selective risk** qua 5 fold.
2. Macro-average không trọng số của **coverage thực tế đạt được** trên test fold (sẽ ≠ mục tiêu).
3. **Bảng 5 giá trị per-fold** của cả risk lẫn coverage, đặt trong phụ lục.

> `ValCalibrated-τ` **không cần thay đổi gì vì lý do khả so sánh giữa các fold**: τ được xác định cục bộ trong từng vòng CV, nên quyết định nhận/từ chối không bao giờ đòi hỏi so sánh điểm số giữa hai model khác nhau. Đây là metric lâm sàng chính của bài và nó miễn nhiễm với vấn đề ở §7.0.

## 3.5 Tái lập

- `folds.json` — commit, không sinh lại.
- Seed cố định: chia fold, khởi tạo model, shuffle dataloader, bootstrap.
- Ghi phiên bản: `torch`, `torchvision`, `scikit-learn`, `numpy`, `iterative-stratification`.

---

# §4. TIỀN XỬ LÝ & CẮT PATCH

## 4.1 Tham số khóa

| Tham số | Giá trị |
|---|---|
| Lề mở rộng | **6%** mỗi phía (tổng +12%) |
| Kích thước đầu ra | **224 × 224** |
| Nội suy | Bilinear |
| **Lưu kích thước & diện tích crop gốc** | **BẮT BUỘC** — biến đồng hành cho §10.3 |

```
x_new = max(0, x − 0.06·w)
y_new = max(0, y − 0.06·h)
w_new = min(W_img, w + 0.12·w)
h_new = min(H_img, h + 0.12·h)
```

Cơ sở: lề 15% đẩy overlap giữa các răng có nhãn từ 15,45% lên 26,15%.

## 4.2 AUGMENTATION — CẤU HÌNH KHÓA CHO THÍ NGHIỆM CHÍNH

> **Cấu hình duy nhất của thí nghiệm chính. Không tìm kiếm, không chọn trên validation, không đổi sau khi train.**

| Phép | Thí nghiệm chính | Ghi chú |
|---|---|---|
| Random rotation ±10° | **BẬT** | Khóa |
| Brightness/contrast jitter nhẹ (±10%) | **BẬT** | Khóa |
| Normalize theo thống kê ImageNet | **BẬT** | Khóa |
| **CLAHE** | **TẮT** | Chỉ ở ablation thứ cấp §9.2 |
| **Horizontal flip** | **TẮT** | Chỉ ở ablation thứ cấp §9.3 |
| **TTA** | **CẤM ở mọi nơi** | Thay đổi phân phối độ tự tin → phá vỡ công bằng baseline |

**Vì sao CLAHE mặc định TẮT:** *cervical burnout chính là vùng thấu quang tương phản thấp.* CLAHE tác động trực tiếp lên đặc trưng ảnh định nghĩa cạm bẫy mà bài lấy làm trung tâm.

**Vì sao horizontal flip mặc định TẮT:** panorama **bất đối xứng về hình học** — ghost image ở phía đối diện và cao hơn, độ phóng đại đổi theo vị trí trong lớp cắt, **vệt kim loại có hướng lan đặc trưng**. Huấn luyện bất biến với lật ngang có thể xóa đúng tín hiệu hướng mà stress-test dựa vào.

---

# §5. MÔ HÌNH & GATE

## 5.1 Backbone — cấu hình khóa

| Hạng mục | Giá trị | Trạng thái |
|---|---|---|
| Kiến trúc | ResNet-50, pretrained ImageNet | KHÓA |
| Loss (thí nghiệm chính) | **Cross-Entropy thuần** | KHÓA |
| Optimizer | AdamW, weight decay 1e-4 | KHÓA |
| **Learning rate** | **1e-4** | **KHÓA — không tìm kiếm** |
| LR schedule | Cosine annealing | KHÓA |
| Batch size | 64 | KHÓA |
| Precision | AMP (FP16) | KHÓA |
| Max epochs | 40 | KHÓA |
| Early stopping | Val loss, patience 8, khôi phục checkpoint tốt nhất | KHÓA (quy tắc khai báo trước, **không** phải phép tìm kiếm) |
| **Số lần train thí nghiệm chính** | **1 per vòng CV → đúng 5 lần** | KHÓA |

**Vì sao LR = 1e-4:** fine-tune backbone pretrained trên tập nhỏ với AdamW; 3e-4 có rủi ro phá hỏng đặc trưng pretrained. Lựa chọn quy ước, **khai báo trước và không kiểm chứng bằng dữ liệu**.

**Về early stopping:** dừng theo val loss là **quy tắc dừng khai báo trước**, không phải phép tìm kiếm siêu tham số — nó không nhân số lần train. Lưu ý: nó làm **số epoch hiệu dụng khác nhau giữa các fold**, đây là một trong những nguồn của hiện tượng ở §7.0.

> **QUY TẮC CÔNG BẰNG:** bảng chính chạy trên **một backbone CE thuần cho TẤT CẢ phương pháp**.

## 5.2 Không gian đặc trưng & manifold

```
z_raw ∈ ℝ²⁰⁴⁸ (penultimate)
   │  PCA fit trên TRAIN FOLDS (hậu nghiệm, KHÔNG đụng backbone)
   ▼
z ∈ ℝᵈ
   │
   ▼
μ_k (4 centroids, train folds)
Σ_shrunk = (1−β*)·Σ_emp + β*·(Tr(Σ_emp)/d)·I     [Ledoit-Wolf, tied]
   │
   ▼
g(x) = − min_{k∈{0..3}} (z − μ_k)ᵀ Σ_shrunk⁻¹ (z − μ_k)
```

**PCA chứ không phải tầng chiếu học được:** tầng chiếu trong backbone làm thay đổi backbone → baseline chạy trên feature khác → phá vỡ công bằng.

**min_k chứ không phải D²_ŷ:** khi model đoán sai ŷ, khoảng cách tới tâm lớp sai vô nghĩa.

**Ghi chú về d:** điều kiện `N_k > d` **KHÔNG áp dụng**. Với tied covariance, Σ ước lượng từ toàn bộ mẫu gộp; μ_k chỉ là trung bình mẫu; Ledoit-Wolf làm Σ luôn khả nghịch.

## 5.3 HAI LOẠI THAM SỐ

> **Nguyên tắc phân loại:** một tham số chỉ được chọn trên validation **khi và chỉ khi** việc thay đổi nó **không đòi hỏi huấn luyện lại**.

### A. Tham số HUẤN LUYỆN — KHÓA HOÀN TOÀN

| Tham số | Giá trị khóa |
|---|---|
| Learning rate | **1e-4** |
| Optimizer / weight decay | AdamW / 1e-4 |
| LR schedule | Cosine |
| Batch size | 64 |
| Max epochs / early stopping | 40 / val loss, patience 8 |
| Augmentation | Cấu hình §4.2 (CLAHE tắt, flip tắt) |
| Loss | Cross-Entropy thuần |
| **`d` (chiều PCA)** | **64 — khóa tiên nghiệm, một giá trị cho cả 5 fold** |

**Tổng số lần train của thí nghiệm chính: 5 — đúng bằng số vòng CV.**

> ### Vì sao `d` nằm ở bảng A (khóa tiên nghiệm) chứ không ở bảng B
>
> **Đây là chỗ sửa một lỗi rò rỉ tinh vi.** Phương án trước đó — "chọn một `d` bằng cách trung bình AURC(validation) qua cả 5 vòng" — **gây rò rỉ**: validation fold của vòng 2 là **F1**, mà F1 chính là **test fold của vòng 1**. Vậy giá trị `d` áp cho test fold của vòng 1 đã được chọn một phần bằng chính dữ liệu của F1. Điều này vi phạm §3.3.
>
> Các cách sửa và lý do loại:
> - *Chọn `d` riêng từng fold*: hết rò rỉ nhưng quay lại 5 lựa chọn phụ thuộc dữ liệu.
> - *Leave-one-round-out*: hết rò rỉ nhưng `d` lại khác nhau giữa các fold.
> - **Khóa tiên nghiệm** ◄ chọn cách này.
>
> **`d = 64` được khóa bằng quy ước, KHÔNG kiểm chứng bằng dữ liệu** — hoàn toàn giống cách xử lý `LR = 1e-4`. Căn cứ quy ước: Σ gộp ước lượng từ ≈2.100 mẫu của 3 train folds, cho tỷ lệ N/d ≈ 33 ở d = 64, so với ≈16 ở d = 128 và ≈8 ở d = 256 — d = 64 giữ ma trận hiệp phương sai co rút ở điều kiện số thoải mái nhất trong lưới.
>
> **Không mất gì:** độ nhạy theo `d` vẫn được báo cáo đầy đủ ở ablation §9.5 (chạy trên vòng CV 1, thứ cấp), nên người đọc vẫn thấy kết quả ở mọi giá trị trong lưới. Khác biệt duy nhất là **kết quả primary không phụ thuộc vào một phép chọn dựa trên dữ liệu**.

### B. Tham số HẬU NGHIỆM — 0 lần train thêm

| Tham số | Phạm vi | Cách chọn |
|---|---|---|
| `α` (trọng số tổ hợp) | 0 → 1, bước 0,05 | **Per-fold**, tối ưu AURC(validation fold) của vòng đó |
| `T` (cho TS baseline) | tối ưu NLL(val) | **Per-fold** |
| `τ` (ngưỡng vận hành) | quét liên tục | **Per-fold**, đạt coverage mục tiêu trên val |

> `α`, `T`, `τ` fit per-fold là **hoàn toàn hợp lệ và không rò rỉ**: mỗi cái khớp trên validation fold của vòng đó rồi áp lên test fold của **chính vòng đó**. Validation fold của một vòng không bao giờ là test fold của chính vòng ấy. **Báo cáo cả 5 giá trị `α*` và `T*`** trong phụ lục.

**Không thêm tham số nào ngoài hai bảng này.**

## 5.4 Hàm tin cậy

```
Φ_S(S(x)) = ECDF của S trên VALIDATION fold của vòng đó   [average rank cho ties]
Φ_M(g(x)) = ECDF của g trên VALIDATION fold của vòng đó   [average rank cho ties]

R_α(x) = α·Φ_S(S(x)) + (1−α)·Φ_M(g(x)),   α ∈ [0,1]
```

Đơn điệu bảo đảm: ∂R/∂Φ_S = α > 0, ∂R/∂Φ_M = 1−α > 0.

**Lưu ý phân tách:** average rank ở đây phục vụ việc **dựng ECDF**, là đối tượng khác với quy tắc tính AURC (§7.1).

## 5.5 SANITY CHECK SỐ 1 — kiểm tra TRONG TỪNG FOLD

> **Khẳng định:** tại α = 1, điểm đề xuất rút gọn thành `R_1(x) = Φ_S(S(x))`.
>
> Vì Φ_S là **phép biến đổi đơn điệu không giảm trong phạm vi một fold**, nó **không thay đổi thứ hạng nào** trong fold đó. Do đó, với **mỗi fold f = 1..5**:
>
> ```
> AURC_f(R_1)  =  AURC_f(MSP)
> ```
>
> **Assertion này chỉ được kiểm tra TRONG TỪNG FOLD.** Không có yêu cầu nào về việc hai phương pháp phải cho cùng bảng xếp hạng sau khi ghép 5 fold — chúng **không** cho cùng bảng xếp hạng, và đó là hành vi đúng, không phải lỗi (xem §7.0).
>
> **Quy trình kiểm tra bắt buộc trong code, lặp cho f = 1..5:**
> 1. Assert: **không có cặp nghịch thế** giữa xếp hạng theo `R_1` và theo `S` trong fold f. Điều kiện cứng; sai ⇒ bug cài đặt percentile.
> 2. Tính `|AURC_f(R_1) − AURC_f(MSP)|`.
> 3. Đếm **số ties mới do Φ_S tạo ra trong fold f** (số cặp mẫu có S khác nhau nhưng Φ_S bằng nhau — xảy ra vì Φ_S là hàm bậc thang khớp trên validation fold).
> 4. Nếu số ties mới = 0 → hai giá trị AURC phải **bằng chính xác**. Lệch ⇒ bug.
> 5. Nếu số ties mới > 0 → độ lệch phải **quy được hoàn toàn về hiện tượng gộp ties đó**.
>
> Ghi kết quả cả 5 fold vào §14. **Không cần thí nghiệm mới** — đây là assertion trong `06_evaluate.py`.
>
> *(Tùy chọn cài đặt: ECDF nội suy tuyến tính thay cho bậc thang sẽ loại bỏ hiện tượng gộp ties ở vùng trong. Nếu dùng, áp **đồng nhất** cho Φ_S và Φ_M và ghi vào DEVIATION LOG.)*

## 5.6 Tên phương pháp — HOÃN

Công thức hiện tại **không còn Energy**, nên tên "EMD-Gate" không khớp. **Không** nhét Energy vào R(x) chỉ để cứu tên. Ablation §9.6 quyết định. **Nguyên tắc:** acronym khai triển thành đúng tập thành phần trong công thức cuối.

---

# §6. BASELINES

Tất cả hậu nghiệm, **cùng một backbone CE thuần**, cùng quy tắc tổng hợp §7.

## 6.1 Bảng chính (primary comparison)

| # | Phương pháp | Nguồn | Train thêm |
|---|---|---|---|
| 1 | MSP | Hendrycks & Gimpel, 2017 | 0 |
| 2 | Temperature Scaling | Guo et al., 2017 | 0 |
| 3 | Mahalanobis (min_k) | Lee et al., 2018 | 0 |
| 4 | **Relative Mahalanobis** | Ren et al., 2021 | 0 |
| 5 | Energy | Liu et al., 2020 | 0 |
| 6 | ViM | Wang et al., 2022 | 0 |
| 7 | **Phương pháp đề xuất** | R_α convex gating | 0 |

## 6.2 MC-Dropout — LOẠI KHỎI BẢNG CHÍNH

> ResNet-50 chuẩn (torchvision) **không chứa lớp Dropout nào**. Hai cách duy nhất để có MC-Dropout đều hỏng:
> - **Chèn dropout rồi train lại** → **backbone khác**, vi phạm ràng buộc chung backbone (§5.1).
> - **Bật dropout chỉ lúc test trên model train không dropout** → sai lệch phân phối kích hoạt → kết quả không hợp lệ.

**Bắt buộc ghi trong bài:**

> *"MC-Dropout was not included because the ResNet-50 backbone contains no dropout layers; introducing them would require a separate training regime and would violate the shared-backbone constraint under which all compared methods are evaluated."*

## 6.3 Relative Mahalanobis — đặc tả đầy đủ

```
FIT (CHỈ TRÊN TRAIN FOLDS — không chạm validation/test fold):
  μ_k, Σ_shrunk   : class-conditional, tied, Ledoit-Wolf          [§5.2]
  μ₀, Σ₀_shrunk   : MỘT Gaussian nền, fit trên TOÀN BỘ feature
                    train folds, BỎ QUA NHÃN, cùng quy trình Ledoit-Wolf

SCORE:
  D²_k(x)  = (z−μ_k)ᵀ Σ_shrunk⁻¹ (z−μ_k)
  D²_0(x)  = (z−μ₀)ᵀ Σ₀_shrunk⁻¹ (z−μ₀)
  RMD_k(x) = D²_k(x) − D²_0(x)
  score(x) = − min_k RMD_k(x)
```

Cùng PCA projection, cùng `d = 64` (§5.3-A), cùng quy trình shrinkage như phương pháp đề xuất. Mọi hệ số co rút β\* ước lượng **chỉ trên train folds**.

**Vì sao bắt buộc:** RMD được thiết kế chính xác để sửa điểm yếu của Mahalanobis thô trên near-distribution — đúng cái Φ_M đang làm. Bỏ nó ⇒ reviewer có lý do yêu cầu bổ sung.

## 6.4 Deep Ensembles — SUPPLEMENTARY REFERENCE

> ### ❌ CẢNH BÁO LEAKAGE
> *"Đã có 5 model từ 5 vòng CV, dùng luôn làm ensemble."*
>
> `model_1` train trên F2,F3,F4. Để ensemble cho test fold F1 cần nhiều model dự đoán F1 — nhưng `model_2` (train F3,F4,F5) **đã thấy F1**, `model_3..5` cũng vậy. **Chỉ `model_1` sạch với F1.**
>
> Đặc biệt nguy hiểm: **mẫu đã ghi nhớ có softmax bão hòa gần 1,0** — bơm độ tự tin giả vào đúng đại lượng bài đang đo.

> ### ✅ ĐÚNG
> Vòng CV 1 duy nhất, K=5 model khác seed trên **cùng bộ train folds của vòng đó** (+4 lần train).
>
> **Vị trí trong bài (khóa):**
> - **Bảng phụ riêng**, tiêu đề *"Supplementary reference: multi-model baseline"*.
> - **KHÔNG** vào bảng primary comparison §6.1.
> - **KHÔNG** thuộc họ so sánh chính §8.2, **không** trong hiệu chỉnh Holm.
> - Ghi chú: *"single CV round only; 5× training cost; not directly comparable to the primary table."*

---

# §7. METRICS & QUY TẮC TỔNG HỢP

## 7.0 Vì sao không ghép điểm thô — tiền đề của toàn bộ §7

AURC là **metric thuần thứ hạng**. Ghép điểm của 5 fold rồi sort chung đòi hỏi một điều kiện mà thiết kế này **không bảo đảm**:

> Điểm `s` do model fold 1 sinh ra và điểm `s` do model fold 3 sinh ra phải ứng với **cùng một xác suất sai**.

Các nguồn làm điều kiện này không thỏa, đều mang tính hệ thống:
- Mỗi fold train trên tập khác → chuẩn trọng số khác → **thang logit khác**.
- **Early stopping (§5.1) dừng ở epoch khác nhau** giữa các fold → **độ sắc softmax khác nhau**.
- PCA basis, μ_k, Σ_shrunk, β\* khác nhau giữa các fold.

Mức độ ảnh hưởng khác nhau theo phương pháp: Energy (không bị chặn, thang tỷ lệ với biên độ logit) nhạy nhất; Mahalanobis/RMD nhạy qua Σ và β\*; ViM nhạy vừa; MSP nhạy ít nhất nhưng **nhạy đúng ở vùng bão hòa** — nơi quyết định AURC; `R_α` ít nhạy nhất vì Φ đã chuẩn hóa hạng theo fold.

**Hệ quả:** ghép điểm thô sẽ đối xử **không đồng đều** giữa các phương pháp, vì phương pháp đề xuất được chuẩn hóa theo fold **do cấu tạo** còn baseline thì không.

> **Gọi tên chính xác:** đây là **cross-fold scale confounding** / **aggregation artifact**. **KHÔNG gọi là "bias"** trừ khi chứng minh được estimator bias theo nghĩa thống kê — điều tài liệu này không làm.

## 7.1 QUY TẮC AURC CHÍNH TẮC — tính TRONG TỪNG FOLD

```
QUY TẮC C-AURC (Block-boundary Risk-Coverage), áp cho MỘT fold

1. Sắp xếp mẫu TRONG TEST FOLD f giảm dần theo điểm tin cậy (float64).
2. Các mẫu có điểm BẰNG NHAU CHÍNH XÁC tạo thành một "tie block".
3. Coverage khả dĩ chỉ nhận giá trị tại BIÊN KHỐI:
      c_b = (số mẫu tích lũy hết khối b) / N_f,   b = 1..B
4. Tại mỗi c_b:  r_b = (số ca sai trong phần được nhận) / (số mẫu được nhận)
5. Đường RC = nội suy TUYẾN TÍNH qua các điểm (c_b, r_b).
6. MỞ RỘNG SANG TRÁI: với c ∈ (0, c_1], đặt r(c) = r_1 (hằng số).
7. AURC_f = tích phân hình thang trên [0, 1].
8. Risk tại coverage mục tiêu c*: nội suy tuyến tính giữa hai biên khối kề;
   nếu c* < c_1 thì r(c*) = r_1 theo bước 6.
9. Đường oracle của E-AURC dùng CHÍNH quy tắc này, trong cùng fold.
```

**Vì sao quy tắc này:** nội suy tuyến tính trong một tie block **bằng đúng** kỳ vọng của selective risk dưới phép phá ties ngẫu nhiên đều. Một quy tắc phục vụ cả hai cách biện minh, kết quả **tất định và tái lập được**.

> ### Vì sao bước 6 (mở rộng sang trái) là bắt buộc
>
> `c_1` = (kích thước tie block đầu tiên)/N_f, và **nó khác nhau giữa các phương pháp**: một phương pháp bão hòa mạnh (ví dụ MSP với nhiều mẫu ở 0,9999) có khối ties đầu tiên lớn, nên `c_1` lớn; một phương pháp có điểm số liên tục có `c_1 ≈ 1/N_f`.
>
> Nếu mỗi phương pháp tích phân trên miền `[c_1, 1]` của riêng nó thì **các AURC được tính trên những miền khác nhau và không so sánh được với nhau** — đúng thứ metric này tồn tại để làm.
>
> Mở rộng hằng số trên `(0, c_1]` đặt **mọi phương pháp trên cùng miền `[0, 1]`**. Đây là quy ước tất định, không thêm giả định nào, và áp y hệt cho đường oracle của E-AURC.

## 7.2 PRIMARY ENDPOINT

$$
\mathrm{AURC_{CV}} \;=\; \frac{1}{5}\sum_{f=1}^{5}\mathrm{AURC}_f
$$

**`AURC_CV` là primary endpoint của bài.** Cùng canonical tie-handling (§7.1) áp trong mọi fold, rồi macro-average **không trọng số**.

### Quy tắc tổng hợp — một quy tắc cho mọi fold-level metric

> **QUY TẮC KHÓA:**
>
> **All fold-level metrics are aggregated by unweighted macro-average across the 5 CV rounds.**
> *(Mọi metric được tính trong phạm vi một fold đều tổng hợp bằng macro-average không trọng số qua 5 vòng CV.)*
>
> Không có ngoại lệ, không có metric nào dùng trọng số theo cỡ fold, và không có metric nào tính bằng cách gộp mẫu của 5 fold trước rồi mới tính.

Danh sách áp dụng: `AURC`, `E-AURC`, `Err-AUROC`, selective risk, `Risk@Coverage` (cả hai phiên bản §3.4), image-level coverage, image-level selective risk, `CB-AURC` (§7.4.2), và mọi metric của phân tích 3-class (§7.4.1).

Hai nhóm metric cần giải thích riêng vì lý do khác nhau:

**① Ratio metrics** — selective risk, coverage, image-level coverage/risk. Đây là **tỷ số của hai đại lượng đếm**, nên **trung bình của 5 tỷ số ≠ tỷ số tính trên tổng gộp**. Hai cách cho hai con số khác nhau và cả hai đều dùng được, nên bắt buộc phải quyết định một cách rồi giữ nhất quán. Chúng tôi chọn **trung bình của 5 tỷ số** (macro-average), để nhất quán với công thức `AURC_CV` ở trên và để mỗi vòng CV có trọng số bằng nhau bất kể cỡ fold lệch chút ít.

**② AUROC** — `Err-AUROC` và AUROC có điều kiện ở §7.5 #2. Đây **không phải tỷ số đếm** mà là thống kê thứ hạng, nên vấn đề "trung bình của tỷ số" ở ① không áp dụng. Nó theo cùng quy tắc tổng hợp vì một lý do khác: **AUROC được tính trên bảng xếp hạng nội bộ một fold**, và bảng xếp hạng của các fold khác nhau không khả so sánh trực tiếp (§7.0) — nên nó phải tính trong fold rồi mới macro-average, y hệt AURC.

> **Bắt buộc kèm bảng 5 giá trị per-fold** trong phụ lục cho mọi metric được macro-average, để người đọc kiểm tra được độ tản mạn giữa các vòng.

### Bảng metric

| Metric | Cách tính | Vai trò |
|---|---|---|
| **`AURC_CV`** | per-fold → macro-average | **PRIMARY** |
| **E-AURC_CV** | per-fold (cùng quy tắc oracle) → macro-average | Primary phụ trợ |
| **Err-AUROC_CV** | per-fold → macro-average | Primary phụ trợ |
| **`ValCalibrated-τ{70,80,90}`** | §3.4 → macro-average, kèm coverage thực đạt | **PRIMARY lâm sàng** |
| **`OracleCov-Risk@{70,80,90}`** | per-fold → macro-average | **Diagnostic** — ghi nhãn rõ |

## 7.3 Coverage cấp ảnh — per-fold rồi macro-average

Gate từ chối **theo từng răng**, nhưng lâm sàng quyết định **theo từng phim**: chỉ cần một răng bị từ chối, cả phim phải chuyển bác sĩ đọc.

```
Trong mỗi test fold f:

I_f = tập ảnh trong fold f có ÍT NHẤT MỘT răng mang nhãn (tổng qua 5 fold: 678 ảnh)

Image-level coverage_f(τ) =
    (số ảnh trong I_f mà MỌI răng có nhãn đều có score ≥ τ) / |I_f|

Image-level selective risk_f(τ) =
    (số ảnh được nhận có ≥1 răng bị phân loại sai) / (số ảnh được nhận)

Sau đó: macro-average không trọng số qua 5 fold.
```

> **Mẫu số là `|I_f|`, không phải tổng số ảnh trong fold.** 27 trong 705 ảnh không có annotation nào. Với ảnh không có răng nào mang nhãn, điều kiện "MỌI răng có nhãn đều có score ≥ τ" đúng **một cách rỗng**: ảnh luôn được nhận ở mọi τ và không bao giờ chứa răng sai, nên sẽ thổi phồng coverage và kéo risk xuống cho mọi phương pháp mà không phản ánh gì về gate. Các ảnh này bị loại khỏi phân tích cấp ảnh — xem §14, 2026-09-29, mục §7.3.

> **CẤM suy diễn giải tích.** Không được viết `1 − 0,8⁵` hay bất kỳ phép tính nhị thức nào. Răng trong cùng một phim **không độc lập**; chiều và độ lớn của sai lệch không biết trước. Con số duy nhất được phép xuất hiện là con số **tính trực tiếp từ predictions**.

Với `ValCalibrated-τ` thì τ đã là per-fold nên an toàn sẵn; với ngưỡng quét thì bắt buộc quét **trong từng fold**.

Báo cáo đường risk–coverage ở **cả hai cấp**, cạnh nhau.

## 7.4 Phân tích 4-class / 3-class / class-balanced

| Phân tích | Vai trò |
|---|---|
| **4-class** | **CHÍNH** — nhãn chính thức DENTEX |
| 3-class (bỏ Impacted) | **PHỤ, khai báo trước** — định nghĩa §7.4.1 |
| Class-balanced selective risk | **PHỤ, khai báo trước** — định nghĩa §7.4.2 |

**Lý do cần class-balanced:** AURC chuẩn bị chi phối bởi tiên nghiệm lớp — Impacted (17%, dễ) tạo khối mẫu ở đầu đường RC, Caries chiếm 62%.

### 7.4.1 Phân tích 3-class — LỌC HẬU NGHIỆM, không huấn luyện lại

> **Quyết định khóa: lọc hậu nghiệm. Không train model 3 lớp.** Ngân sách 12 lần train ở §13.2 không đổi.

```
Mô hình, gate, Φ_S, Φ_M, α*, T*, τ:  GIỮ NGUYÊN — tất cả vẫn khớp trên 4 lớp.
Đầu dự đoán:                          GIỮ NGUYÊN 4 lớp.

Tập đánh giá:  chỉ giữ mẫu có NHÃN THẬT ≠ Impacted.
Định nghĩa lỗi: prediction ≠ nhãn thật, KỂ CẢ khi prediction = Impacted.
Tính C-AURC (§7.1) TRÊN TẬP ĐÃ LỌC, trong từng fold → macro-average.
```

**Hai điểm bắt buộc làm đúng:**

1. **Lọc theo NHÃN THẬT, không theo nhãn dự đoán.** Lọc theo dự đoán là điều kiện hóa trên kết quả — sẽ loại bỏ đúng những ca model nhầm sang Impacted, tức là giấu đi một phần lỗi.
2. **Mẫu bị đoán nhầm thành Impacted vẫn tính là lỗi.** Model không được "miễn" loại lỗi đó chỉ vì lớp Impacted bị loại khỏi tập đánh giá.

**Phân tích này trả lời:** *khi bỏ lớp dễ ra khỏi phép đo, khác biệt giữa các phương pháp có còn không?*
**Nó KHÔNG trả lời:** *hệ thống hoạt động ra sao nếu được huấn luyện mà không cần phân biệt Impacted* — câu đó cần huấn luyện lại và **nằm ngoài phạm vi**. Phải ghi rõ điều này khi báo cáo.

### 7.4.2 Class-balanced selective risk — định nghĩa

> Ba cách hiểu thông dụng cho ba con số khác nhau. **Khóa đúng một cách dưới đây.**

Với test fold `f`, gọi `n_{k,f}` là số mẫu lớp `k` trong fold đó và `K = 4`. Gán cho mỗi mẫu `i` một **trọng số cố định**:

$$
w_i \;=\; \frac{1}{K \cdot n_{y_i, f}}
$$

Với tập được chấp nhận `A` tại một coverage bất kỳ:

$$
\mathrm{Risk}_{\mathrm{CB}}(A) \;=\; \frac{\sum_{i \in A} w_i \cdot \mathbb{1}[\hat{y}_i \neq y_i]}{\sum_{i \in A} w_i}
$$

**Ba tính chất khiến định nghĩa này được chọn:**

- Khi `A` = toàn bộ fold, nó rút gọn **đúng bằng** trung bình không trọng số của tỷ lệ lỗi từng lớp (balanced error rate).
- **Không bao giờ chia cho 0 theo lớp**: lớp không có mẫu nào được chấp nhận chỉ đóng góp 0 vào cả tử và mẫu.
- Nó **thay thế đúng một thành phần** trong quy tắc C-AURC (§7.1 bước 4) — phần còn lại của quy tắc giữ nguyên, nên `CB-AURC` tính bằng chính hàm đó.

> **Coverage vẫn KHÔNG trọng số.** Coverage là đại lượng vận hành — tỷ lệ ca mà AI xử lý — nên đếm theo mẫu, không theo trọng số lớp. Chỉ **tử số rủi ro** được cân bằng lớp.

> **Lưu ý cỡ mẫu:** phân tích riêng cho Periapical ở mức per-fold chỉ có ≈31 mẫu (§2.2). Kết quả per-fold cho lớp này **phải kèm CI** và nhiều khả năng không kết luận được. Phân tích lớp hiếm có ý nghĩa nằm ở **nhánh secondary §7.6** (n = 157; trước đây ghi 158 — xem §14).

## 7.5 Chẩn đoán giá trị gia tăng của Φ_M

> **Không dùng ngưỡng Spearman.** Tương quan hạng toàn cục bị chi phối bởi khối đa số ca dễ. Hai điểm số có thể tương quan 0,95 mà vẫn bất đồng đúng ở các ca sai tự tin cao — chính các ca đó quyết định AURC.

| # | Chẩn đoán | Nhánh |
|---|---|---|
| 1 | **ΔAURC + CI bootstrap lồng ghép cặp** giữa α=1 / α=0 / α=α\*, tính trên **`AURC_CV`** | **PRIMARY** |
| 2 | **AUROC có điều kiện ở decile tự tin cao nhất** — định nghĩa đầy đủ ngay dưới | **SECONDARY (§7.6)** |
| 3 | **Scatter (Φ_S, Φ_M) tô màu đúng/sai** | Minh họa |

> **Vì sao chẩn đoán #2 nằm ở nhánh secondary:** trong một fold, decile cao nhất chỉ còn ≈70 patch — không đủ để đọc AUROC có điều kiện. Đây là con số **đo trực tiếp cơ chế mà bài báo tuyên bố**, nên nó cần cỡ mẫu của nhánh ghép chuẩn hóa. Vị trí này **khóa trước**, không chọn sau khi thấy kết quả.

### Định nghĩa đầy đủ của chẩn đoán #2

```
Miền tính:   nhánh SECONDARY §7.6 (đã chuẩn hóa hạng theo fold, ghép 3.523 mẫu)

Bước 1.  Xếp hạng toàn bộ mẫu theo  Φ_S đã chuẩn hóa  — TÍN HIỆU SOFTMAX,
         không phải Φ_M, không phải R_α.
Bước 2.  Lấy DECILE CAO NHẤT theo thứ hạng đó (≈352 mẫu).
Bước 3.  Trong riêng nhóm đó, tính AUROC của  Φ_M đã chuẩn hóa
         để tách  {đoán đúng}  vs  {đoán sai}.
Bước 4.  Báo cáo kèm: cỡ mẫu decile, SỐ CA SAI trong decile, và CI bootstrap.
```

**Vì sao decile lấy theo Φ_S chứ không theo Φ_M hay R_α:** câu hỏi cơ chế của bài là *"trong những ca mà softmax tự tin nhất, tín hiệu manifold có tách được ca sai không?"*. Lấy decile theo Φ_M sẽ điều kiện hóa trên chính tín hiệu đang được đánh giá; lấy theo R_α sẽ trộn hai tín hiệu và làm câu hỏi mất nghĩa.

> **Nếu số ca sai trong decile quá nhỏ, AUROC không đọc được.** Khi đó báo cáo cỡ mẫu và nói thẳng là không ước lượng được — **không** hạ ngưỡng decile xuống để có thêm mẫu, vì đó là thay đổi phân tích dựa trên kết quả.

Spearman báo cáo như số liệu mô tả trong phụ lục.

## 7.6 NHÁNH SECONDARY — Fold-wise rank-normalized pooled analysis

### Định nghĩa

```
Với MỖI phương pháp m (bao gồm TOÀN BỘ baseline VÀ phương pháp đề xuất):

  Bước 1. Với mỗi vòng CV f, gọi s_m,f là điểm cuối cùng của phương pháp m
          trong vòng đó.
  Bước 2. Φ_m,f := ECDF của s_m,f trên VALIDATION fold của vòng f
          (average rank cho ties).
  Bước 3. Với mỗi mẫu x thuộc test fold f:   s̃_m(x) = Φ_m,f( s_m,f(x) )
  Bước 4. Ghép s̃_m của cả 5 fold → một bảng xếp hạng chung
          → tính C-AURC (§7.1) trên toàn bộ 3.523 mẫu.
```

**Áp dụng ĐỒNG NHẤT cho mọi phương pháp**, kể cả phương pháp đề xuất — tức `R_α` cũng đi qua bước 2–3 (`Φ_R,f(R_α)`), để không phương pháp nào được đối xử khác.

### Tên gọi bắt buộc

Trong mọi bảng, hình và câu văn, nhánh này được gọi là:

> **fold-wise rank-normalized pooled analysis**

và được đặt ở **secondary / sensitivity analysis**, **không phải primary endpoint**.

### Phát biểu chuẩn về bản chất của phép chuẩn hóa

> *"ECDF normalization does not change the within-fold ranking or within-fold AURC, but it changes the cross-fold aggregation rule when scores are pooled."*

Đây là câu **bắt buộc** dùng khi mô tả nhánh này. Hai mệnh đề trong đó khác nhau và không được gộp:
- **Trong fold:** ECDF là phép biến đổi đơn điệu → thứ hạng và AURC_f **không đổi** (tới sai khác do gộp ties).
- **Khi ghép:** nó **thay đổi quy tắc tổng hợp liên fold** — đó chính là mục đích của nó.

> ⚠️ **CẤM mô tả phép chuẩn hóa này như một cải tiến của baseline.** Nó là **aggregation device** để làm điểm số khả so sánh giữa các fold được huấn luyện độc lập. "Energy (Liu et al., 2020)" trong bảng secondary phải ghi kèm *"fold-wise rank-normalised"* trong caption.

### Nhánh này phục vụ những gì

| Phân tích | Vì sao ở đây |
|---|---|
| Lớp hiếm (Periapical, n = 157; trước đây ghi 158 — xem §14) | per-fold chỉ có ≈31 |
| Nhóm cạm bẫy (§11.1) | per-fold quá nhỏ — xem §11.1 |
| AUROC có điều kiện ở decile tự tin cao (§7.5 #2) | per-fold chỉ ≈70 patch |

### Sanity check phụ của nhánh này

Theo cấu tạo, `α=1` sau chuẩn hóa trùng với MSP sau chuẩn hóa (cả hai đều là ECDF của một biến đổi đơn điệu của S), nên:

```
C-AURC_pooled-norm(α=1)  ≈  C-AURC_pooled-norm(MSP)
```
tới sai khác do gộp ties. Kiểm tra và ghi vào §14.

### Ranh giới sử dụng — khóa

> **CẤM dùng `|AURC_raw − AURC_norm|` (hay bất kỳ so sánh raw-vs-normalized nào) làm quy tắc quyết định có giữ raw pooling hay không.** Làm vậy khiến lựa chọn tổng hợp **phụ thuộc dữ liệu** — đúng thứ pre-registration tồn tại để loại trừ.
>
> Quy tắc tổng hợp đã khóa tại §7.2 (primary) và §7.6 (secondary). Nếu vẫn muốn tính raw pooling, nó **chỉ được báo cáo như sensitivity analysis trong phụ lục**, không bao giờ là căn cứ để đổi protocol.

---

# §8. THỐNG KÊ

## 8.1 Bootstrap lồng, giữ cấu trúc fold

> ### ⚠️ Đơn vị tính metric ≠ đơn vị lấy mẫu lại
>
> **`AURC` là một tooth-level metric** — nó được tính trên từng patch răng, và mỗi patch là một quan sát trong đường risk–coverage.
>
> **Nhưng đơn vị resample của bootstrap là ẢNH PANORAMA (cluster), không phải patch răng.**
>
> Lý do: các răng trong cùng một phim **không độc lập** — chúng chia sẻ bệnh nhân, máy chụp, liều tia, chất lượng phim và mức nhiễu kim loại. Resample ở cấp patch coi ~3.523 patch như 3.523 quan sát độc lập, trong khi số đơn vị độc lập thực tế gần với **678 ảnh** hơn — 27 trong 705 ảnh không có annotation nào nên không đóng góp patch nào cho metric cấp răng (§2.1, §7.3). Hệ quả là CI **hẹp giả tạo** và mọi tuyên bố ý nghĩa thống kê đều không đáng tin.
>
> Vì vậy: khi một ảnh được lấy vào mẫu bootstrap, **toàn bộ patch của ảnh đó được lấy theo** — nguyên khối, không tách rời.
>
> Cách này (cluster bootstrap) xử lý phụ thuộc **phi tham số**: không cần biết ICC, không cần giả định cấu trúc tương quan.

```
Lặp B = 1.000 lần:
  Với mỗi fold f = 1..5:
      (i)  CẤP RĂNG: resample CÓ HOÀN LẠI |T_f| ảnh từ T_f
           (T_f = TOÀN BỘ ảnh trong test fold f, kể cả ảnh không có annotation)
           (mọi patch của một ảnh được lấy theo ảnh đó — cluster)
           tính lại AURC_f và mọi metric cấp răng trên tập resample
      (ii) CẤP ẢNH: resample CÓ HOÀN LẠI |I_f| ảnh từ I_f
           (I_f = ảnh trong test fold f có ≥1 răng mang nhãn, định nghĩa ở §7.3)
           tính lại image-level coverage_f / risk_f trên tập resample
  AURC_CV^(b) = (1/5) · Σ_f AURC_f^(b)      (tương tự cho metric cấp ảnh)

CI 95% = phân vị 2,5 và 97,5 của phân phối AURC_CV^(b).
```

**Tập ảnh được resample — khóa theo họ metric:**

| Họ metric | Tập resample trong fold f | Tính chất |
|---|---|---|
| **Cấp ảnh** (§7.3: image-level coverage, image-level selective risk) | **`I_f`** — chỉ ảnh có ≥1 răng mang nhãn | **Bắt buộc.** Metric định nghĩa trên `I_f`; bốc từ toàn bộ fold sẽ đưa 27 ảnh rỗng quay lại qua bootstrap và tái lập đúng thiên lệch rỗng-đúng mà §7.3 đã loại. Điểm ước lượng và CI phải cùng một tổng thể |
| **Cấp răng** (`AURC`, `E-AURC`, `Err-AUROC`, selective risk, `Risk@Coverage`, `CB-AURC`, 3-class, và mọi metric patch-level) | **`T_f`** — toàn bộ ảnh trong test fold, **kể cả ảnh rỗng** | **Lựa chọn, đã khóa.** Bốc từ `T_f` giữ được biến thiên "có phim không mang tổn thương nào" — đúng tổng thể phim mà bài ước lượng. Ảnh rỗng khi được bốc đóng góp 0 patch; đó là hành vi đúng, không phải lãng phí. (Bốc từ `I_f` cũng bảo vệ được; không dùng.) |

Hai lần bốc (i) và (ii) là **độc lập** trong cùng lần lặp `b`. Ràng buộc ghép cặp áp **trong từng họ metric**: mọi phương pháp dùng cùng tập bốc (i) cho metric cấp răng và cùng tập bốc (ii) cho metric cấp ảnh. Xem §14, 2026-09-29, mục §8.1.

**Ràng buộc bắt buộc:**

| Quy định | Lý do |
|---|---|
| Resample ở **cấp ảnh** — tập ảnh theo họ metric: `T_f` cho cấp răng, `I_f` cho cấp ảnh (bảng trên) | Patch cùng phim không độc lập (chia sẻ bệnh nhân, máy chụp, liều tia, mức nhiễu) |
| **Không** resample patch như quan sát độc lập | CI hẹp giả tạo |
| **Không** resample fold | Fold không phải đơn vị lấy mẫu; §8.3 |
| **Giữ cấu trúc fold** — resample trong từng fold | Giữ đúng cấu trúc CV |
| **Giữ paired predictions giữa các phương pháp** | Mọi phương pháp chấm cùng tập mẫu resample trong cùng lần lặp `b` |

Với `ΔAURC_CV` giữa hai phương pháp: tính hiệu **trong cùng lần lặp `b`**, rồi lấy phân vị của phân phối hiệu — đây là CI ghép cặp.

**Xử lý lần lặp suy biến (bắt buộc, để tránh hành vi không xác định):** khi bootstrap một **nhóm con nhỏ** (§11.1), một lần lặp có thể cho ra tập rỗng hoặc tập không có ca sai nào, khiến selective risk không xác định. Quy tắc: **bỏ lần lặp đó và đếm số lần bỏ**; nếu số lần bỏ vượt 5% tổng số lần lặp thì **không báo cáo CI cho nhóm con đó**, chỉ báo cáo cỡ mẫu và nói rõ là không ước lượng được. Không thay thế bằng giá trị mặc định.

**Về design effect:** hệ số `1 + (m−1)·ICC` phụ thuộc ICC thực tế, **chưa biết**. ICC sẽ được **ước lượng từ dữ liệu** và báo cáo như số liệu mô tả. **Không** dùng con số giả định để tuyên bố về power. Cluster bootstrap không cần biết ICC — nó xử lý phụ thuộc phi tham số.

## 8.2 So sánh khai báo trước

| Loại | Nội dung | Đại lượng |
|---|---|---|
| **Chính (primary)** | (a) Đề xuất vs **MSP** · (b) Đề xuất vs **ViM** | **`ΔAURC_CV`**, CI từ §8.1 |
| **Phụ (secondary)** | Các phương pháp còn lại trong §6.1 | Mô tả, không tuyên bố ý nghĩa thống kê |
| **Ngoài họ so sánh** | **Deep Ensembles (§6.4)** | Bảng phụ, không kiểm định, không Holm |
| Hiệu chỉnh đa so sánh | **Holm**, nếu tuyên bố ý nghĩa trên toàn bảng §6.1 | |

## 8.3 Điều CẤM

> Không dùng 5 giá trị `AURC_f` làm 5 quan sát độc lập để chạy t-test hoặc để dựng CI. Tập train chồng lấn 75%; giả định độc lập sai (Dietterich, 1998). **CI phải đến từ bootstrap lồng ở §8.1**, nơi đơn vị resample là ảnh chứ không phải fold.

## 8.4 Bản chất đại lượng ước lượng

`AURC_CV` là **trung bình của năm giá trị AURC, mỗi giá trị thuộc về đúng MỘT mô hình và được tính hoàn toàn trong phạm vi dữ liệu mà mô hình đó chưa từng thấy.**

Nó ước lượng cho **AURC kỳ vọng của quy trình huấn luyện** dưới phép lấy mẫu tập huấn luyện. Cách tính này:

- **Không giả định khả so sánh liên model nào** — đây là điểm mạnh so với mọi phương án ghép.
- Bắt được **phương sai lấy mẫu** (bệnh nhân nào rơi vào đâu) qua bootstrap §8.1.
- **Không** bắt được **phương sai huấn luyện** (seed, khởi tạo) — chỉ có 5 lần train, mỗi fold một lần. **Khai báo giới hạn này trong Limitations.**

Nhánh secondary §7.6 ước lượng một đại lượng **khác**: AURC của một bảng xếp hạng chung sau khi các điểm số được đưa về thang phân vị nội bộ fold. Hai đại lượng **không thay thế cho nhau** và phải được trình bày riêng.

---

# §9. ABLATION — TẤT CẢ LÀ THỨ CẤP

> **Không kết quả ablation nào được dùng làm con số headline.** Bảng chính luôn là backbone CE thuần với cấu hình khóa §4.2 + §5.3-A.

> ### Phạm vi fold của mỗi ablation — quyết định theo chi phí, khóa trước
>
> Ablation **không cần train thêm** chạy được trên **cả 5 vòng CV** với chi phí gần bằng 0 (chỉ tính lại điểm từ feature đã lưu), nên chúng báo cáo `AURC_CV` như phần chính.
> Ablation **cần train thêm** chỉ chạy trên **vòng CV 1** để tiết kiệm compute, và báo cáo `AURC_1` — **không macro-average**, phải ghi rõ "single CV round" khi trình bày.

| # | Ablation | Cấu hình | Train thêm | Phạm vi |
|---|---|---|---|---|
| 9.1 | Phân rã thành phần | α=1 / α=0 / α=α\* | 0 | **5 fold → `AURC_CV`** |
| 9.2 | **CLAHE** | tắt (chính) vs bật | 1 | Vòng CV 1 |
| 9.3 | **Horizontal flip** | tắt (chính) vs bật | 1 | Vòng CV 1 |
| 9.4 | **Center Loss** | CE vs CE+λ·Center — **chạy lại TOÀN BỘ baseline** | 1 | Vòng CV 1 |
| 9.5 | **Độ nhạy theo `d`** | {32, **64**, 128, 256} — **64** là giá trị khóa của bài | 0 | **5 fold → `AURC_CV`** |
| 9.6 | Energy làm thành phần thứ 3 | R = α·Φ_S + β·Φ_M + γ·Φ_E | 0 | **5 fold → `AURC_CV`** |

**§9.1 chính là chẩn đoán §7.5 #1** — cùng một phân tích, không phải hai lần tính. Trình bày một lần, tham chiếu chéo giữa hai mục.

**§9.2 và §9.3 là ablation mô tả, không phải phép chọn cấu hình.** Kết quả của chúng **không** được dùng để đổi cấu hình chính (§4.2 đã khóa). Chúng trả lời *"nếu bật thì sao"*, và câu trả lời đi vào Discussion.

**§9.5 là ablation ĐỘ NHẠY, không phải phép chọn `d`.** Giá trị `d = 64` đã khóa tiên nghiệm ở §5.3-A và **không được đổi** dựa trên kết quả của §9.5. Ablation này chỉ để người đọc thấy kết quả biến thiên ra sao trong lưới.

## 9.4 Quy tắc Center Loss

> Center Loss **nén cụm đặc trưng chặt lại**, có lợi bất đối xứng cho các phương pháp dựa trên khoảng cách.
>
> - ❌ **CẤM:** train phương pháp đề xuất bằng CE+Center rồi so với baseline train bằng CE thuần.
> - ✅ **BẮT BUỘC:** nếu chạy, **mọi phương pháp §6.1 được chạy lại** trên backbone Center Loss, trình bày thành bảng **riêng biệt**.
> - Nếu không đủ compute chạy lại toàn bộ → **bỏ hẳn**, không chạy một nửa.

---

# §10. KIỂM TRA CHẶN ĐƯỜNG & DIAGNOSTIC

> ### ⚠️ Bốn kiểm tra này chạy ở HAI THỜI ĐIỂM KHÁC NHAU
>
> v5 xếp cả bốn vào `00_sanity_checks.py` "trước khi cắt patch". **Điều đó không thực hiện được**: §10.3 cần `g(x)` (đòi hỏi backbone đã train và manifold đã tính) và §10.4 cần điểm số của mọi phương pháp (đòi hỏi đã chấm điểm xong). Chỉ §10.1 và §10.2 chạy được trước khi cắt patch.
>
> | Kiểm tra | Thời điểm | Script | Vai trò |
> |---|---|---|---|
> | §10.1 Phân bố lớp theo fold | Sau khi có `folds.json`, **trước khi cắt patch** | `00_sanity_checks.py` | Chặn đường |
> | §10.2 Discriminator giữa hai subset | **Trước khi cắt patch** (dùng ảnh nguyên) | `00_sanity_checks.py` | Chặn đường, **có điều kiện** |
> | §10.3 Hồi quy `g(x)` ~ diện tích crop | **Sau `04_compute_manifold.py`** | `06_evaluate.py` | Diagnostic báo cáo |
> | §10.4 Đếm unique score | **Sau khi chấm điểm mọi phương pháp** | `06_evaluate.py` | Diagnostic báo cáo |
>
> §10.1 và §10.2 là **chặn đường** — kết quả có thể đổi thiết kế xuống dòng. §10.3 và §10.4 là **diagnostic báo cáo** — không bao giờ đổi mô hình hay protocol (§10.3).

## 10.1 Phân bố lớp theo fold

Báo cáo phân bố 4 lớp trên mỗi fold sau iterative stratification. **Chỉ để ghi nhận.** Không chia lại fold dựa trên kết quả này (§3.1).

## 10.2 Discriminator giữa hai subset — diễn giải đúng

> **Có điều kiện:** kiểm tra này chỉ cần thiết nếu **thí nghiệm phụ §12.1 (Unlabeled Tooth Pool) được thực hiện**, hoặc nếu patch pilot lấy theo phương án (a) ở §11.2. Nếu bỏ cả hai thì §10.2 không cần chạy.

Train phân loại nhị phân phân biệt ảnh thuộc subset `disease` (705) hay `quadrant_enumeration` (634).

| Kết quả | Diễn giải **đúng** |
|---|---|
| AUC gần 0,5 | **"Discriminator này, với kiến trúc và cỡ mẫu này, không phát hiện được dịch chuyển phân biệt được giữa hai subset."** |
| AUC cao | **Có dịch chuyển phân biệt được** → mọi kết quả trên pool khai báo là **cross-subset distribution shift**; tuyệt đối không gọi là OOD |

> ⚠️ **Diễn giải bị CẤM:** AUC ≈ 0,5 **không** chứng minh hai phân phối giống nhau. Đây là **vắng mặt bằng chứng, không phải bằng chứng vắng mặt** — kết quả bị giới hạn bởi năng lực của discriminator, cỡ mẫu, và loại dịch chuyển nó có thể bắt. Không viết "hai subset có phân phối tương đương".

## 10.3 Hồi quy g(x) ~ kích thước / diện tích crop gốc — CHỈ LÀ DIAGNOSTIC

**Vấn đề:** patch răng hàm và răng cửa có kích thước gốc chênh nhau nhiều lần; cả hai resize về 224×224 → thông tin độ phóng đại bị xóa. Kích thước tương quan với loại răng, loại răng tương quan với lớp. Rủi ro: manifold một phần mã hóa **kích thước răng** chứ không phải bệnh lý.

> **QUY TẮC:** đây là **diagnostic báo cáo trong bài**, **KHÔNG** phải trigger để thay đổi mô hình.

**Dự phòng khai báo trước** (một *phân tích* bổ sung): nếu tương quan mạnh, báo cáo thêm **AURC phân tầng theo tam phân vị diện tích crop**. Mô hình không đổi.

## 10.4 Đếm unique score

Số giá trị điểm phân biệt của mỗi phương pháp, **tính trong từng fold** → phụ lục.

---

# §11. STRESS-TEST LÂM SÀNG

## 11.1 Thí nghiệm chính — và giới hạn cỡ mẫu phải khai báo trước

Nhóm con trong **clinical annotation cohort** (§0), gán nhãn ba loại cạm bẫy:

1. Thấu quang cổ răng sinh lý (*cervical burnout*)
2. Vệt cản quang / nhiễu tán xạ kim loại (amalgam, mão kim loại)
3. Vùng cản quang giải phẫu chồng lấn (lỗ cằm, xoang hàm trên)

**Quy tắc tổng hợp:**

| Nhánh | Cách tính | Vai trò |
|---|---|---|
| Primary | AURC và Risk@cov **per-fold** cho từng nhóm → macro-average | Nhất quán với §7.2 |
| **Secondary (§7.6)** | Ghép có chuẩn hóa hạng theo fold, áp đồng nhất mọi phương pháp | **Nơi kết quả subgroup thực sự đọc được** |

> ### ⚠️ Khai báo trước về cỡ mẫu — không để lộ ra lúc viết bản thảo
>
> Cỡ mẫu nhóm con trong một fold là `prevalence × ≈705 patch`. **Prevalence của từng loại cạm bẫy chưa biết** và chỉ xác định được sau khi gán nhãn (§11.2) — **không ước đoán con số ở đây**.
>
> Điều biết chắc: với một loại cạm bẫy có prevalence thấp, per-fold sẽ còn vài chục patch hoặc ít hơn, và **phân tích per-fold cho nhóm đó nhiều khả năng không kết luận được gì**. Vì vậy:
> - Kết quả subgroup **primary** vẫn báo cáo per-fold → macro-average, **luôn kèm CI**, và nếu CI quá rộng thì **nói thẳng là không kết luận được**, không diễn giải xu hướng từ điểm ước lượng.
> - Kết quả subgroup có ý nghĩa thực tế nằm ở **nhánh secondary §7.6**, và bài báo phải trình bày đúng như vậy.
> - **Limitations bắt buộc ghi nhận** điều này.
>
> Đây là **subgroup analysis lồng trong OOF prediction set**, không phải tập đánh giá độc lập.

## 11.2 Protocol gán nhãn — khóa

```
BƯỚC 1.  Viết annotation guideline v1
         (định nghĩa cervical burnout; ranh giới phân biệt với sâu cổ răng thật;
          tiêu chí vệt kim loại; vùng chồng lấn giải phẫu)

BƯỚC 2.  Hiệu chỉnh guideline trên 15 patch PILOT
         ⚠️ NGUỒN PILOT — ưu tiên theo thứ tự:
            (a) 15 patch cắt từ ảnh thuộc subset `quadrant_enumeration`
                (KHÔNG nằm trong 705 ảnh) → OOF prediction set nguyên vẹn.  ◄ ƯU TIÊN
            (b) Nếu (a) bất khả thi: chọn 15 patch từ trong 705 ảnh,
                ghi danh sách, LOẠI TRỪ VĨNH VIỄN khỏi clinical annotation
                cohort và khỏi mọi kết quả subgroup §11.1.
                (Chúng vẫn ở lại trong OOF prediction set cho đánh giá mô hình
                 chính, vì đánh giá đó không dùng nhãn cạm bẫy.)

BƯỚC 3.  🔒 KHÓA GUIDELINE  ← không sửa sau bước này

BƯỚC 4.  Gán nhãn mù trên CLINICAL ANNOTATION COHORT (không lọc trước)
         → MÙ với output của model

BƯỚC 5.  Sau ≥3 TUẦN: đọc lại ngẫu nhiên 100 mục
         → mù với nhãn lần 1, thứ tự xáo lại, mù với output model

BƯỚC 6.  Báo cáo intra-rater κ kèm CI 95%
```

> **Vì sao nguồn pilot quan trọng:** trong 5-fold CV **mọi patch đều nằm trong OOF prediction set** — không có patch nào "chỉ thuộc train". Nếu guideline được hiệu chỉnh bằng patch nằm trong cohort đánh giá, tiêu chí gán nhãn đã được điều chỉnh theo chính dữ liệu dùng để báo cáo — rò rỉ ở tầng nhãn.

## 11.3 Ba lớp mù

| Mù với | Lý do |
|---|---|
| Nhãn lần một | Giảm hiệu ứng nhớ |
| Thứ tự trình bày | Giảm hiệu ứng vị trí |
| **Output của model** | **Nếu bác sĩ gán nhãn cạm bẫy trên ca model đã đoán sai, tập nhãn thành hàm của model → lập luận vòng tròn** |

## 11.4 Tuyên bố độ tin cậy — dùng đúng từ

| Tình huống | Được gọi là | KHÔNG được gọi là |
|---|---|---|
| 1 bác sĩ, đọc 2 lần | **intra-rater reliability** | inter-rater reliability |
| 2 người đọc độc lập | **inter-rater reliability** | |

Thống kê là Cohen's κ trong cả hai; **cái thay đổi là tuyên bố, không phải công thức**.

## 11.5 Người đọc thứ hai cho nhãn artifact

Sự hiện diện của **vệt kim loại** là phán đoán thị giác, không cần chuyên môn RHM — người đọc thứ hai được huấn luyện ngắn là đủ.

→ Có **inter-rater κ thật cho nhãn artifact**, trong khi bác sĩ độc quyền phụ trách phán đoán bệnh lý (intra-rater).

---

# §12. THÍ NGHIỆM PHỤ & GIỚI HẠN CLAIM

## 12.1 Unlabeled Tooth Pool — chỉ làm nếu có nhãn chuyên gia

### Thuật ngữ

| Thuật ngữ | Áp dụng? | Lý do |
|---|---|---|
| **Unlabeled** | ✅ | Điều duy nhất biết chắc |
| **Unknown** | ✅ | Chính xác nhất |
| **Near-distribution** | ✅ | Cùng modality, cùng giải phẫu |
| Healthy | ❌ | Cần annotation. Không có |
| Negative | ❌ | Cần annotation. Không có |
| **OOD** | ❌ | **Răng không nhãn hoàn toàn có thể đang mang bệnh** |

### Vì sao abstention rate đơn thuần KHÔNG phải metric

Nó chỉ đo **độ dịch chuyển phân phối điểm tin cậy**, không đo tính đúng đắn của quyết định nào — vì không quyết định nào có đáp án.

**Failure case — reject 100% pool:** không tốt, không xấu, **vô nghĩa**. Ba khả năng không phân biệt được: (a) gate nhạy với răng bất thường; (b) gate phát hiện khác biệt tiền xử lý giữa subset; (c) mọi răng ngoài tập annotated đều ngoài manifold một cách tầm thường.

Tệ hơn: nếu pool có răng sâu ngà thật và model nói "Deep Caries" tự tin cao, **từ chối là quyết định SAI** — nhưng metric lại tính là điểm cộng.

### Điều kiện để pool dùng được — và cách đặt tên lớp thứ 5

Bác sĩ đọc **mẫu ngẫu nhiên ~300 răng**, chọn một trong ba:

| Nhãn | Diễn đạt bắt buộc |
|---|---|
| A | **`no DENTEX-category finding observed`** |
| B | có bệnh lý thuộc 4 lớp (ghi lớp) |
| C | không xác định được |

> ⚠️ **Nhãn A KHÔNG có nghĩa là "răng khỏe mạnh".** Nó là phát biểu về **quan sát của người đọc trong phạm vi 4 lớp DENTEX**. Răng mang nhãn A vẫn có thể có bệnh nha chu, phục hồi cũ, tổn thương không thuộc 4 lớp, hoặc tổn thương không nhìn thấy trên phim. **Không bao giờ viết "healthy" cho lớp này.**

Khi đó pool thành tập đánh giá có lớp thứ 5:
- răng nhãn A → hành động đúng là **abstain**
- răng nhãn B → hành động đúng là **phân loại đúng**
- răng nhãn C → loại khỏi phân tích, báo cáo số lượng

Kèm control: tỷ lệ từ chối trên răng **có nhãn bệnh** cùng ảnh, và so với điểm số ngây thơ (độ sắc nét, kích thước patch).

### 🚩 Phương án BỊ LOẠI: same-image unlabeled teeth

Mỗi phim có ~28–32 răng nhưng chỉ ~5 răng có hộp. Không có bounding box cho phần còn lại → phải train tooth detector → thêm giai đoạn object detection mà thiết kế đã cố ý tránh. **SCOPE CREEP — LOẠI.**

## 12.2 🚩 External dataset — LOẠI

| Lý do | Chi tiết |
|---|---|
| Không có nhãn tương thích | Không validate được bộ phân loại |
| Overclaim | "External validation cho selective pathology classification" sẽ bị reviewer bắt |
| Scope | ≈ 2–3 tuần |
| **Động cơ sai** | Xuất phát từ mong muốn nâng tier tạp chí, không từ câu hỏi nghiên cứu |

**Thay thế:** Limitations ghi trung thực — *"evaluated on a single publicly released dataset; the number of contributing institutions and scanner models were not independently verified in this work"*.

## 12.3 Danh sách điều bài báo KHÔNG tuyên bố

- ❌ Phát hiện false positive trên răng lành
- ❌ Phát hiện OOD
- ❌ Có external validation
- ❌ Đề xuất cơ chế ước lượng bất định mới về mặt toán học
- ❌ Giảm tải lâm sàng, trừ khi image-level coverage (§7.3) chứng minh **bằng số đo trực tiếp**
- ❌ Inter-rater reliability, nếu chỉ một bác sĩ đọc bệnh lý
- ❌ "Confirmed disease" · ❌ "Single-center" · ❌ "Healthy" cho răng không có nhãn bệnh
- ❌ Hai subset có phân phối tương đương (§10.2)
- ❌ So sánh được với bảng xếp hạng DENTEX challenge
- ❌ Deep Ensembles nằm trong primary comparison
- ❌ **Báo cáo pooled RAW AURC như primary endpoint** — primary là `AURC_CV` (§7.2)
- ❌ **Mô tả fold-wise rank normalization như một cải tiến của baseline** — nó là aggregation device (§7.6)
- ❌ **Dùng `raw-vs-normalized AURC gap` làm quy tắc quyết định tổng hợp** — nếu tính, chỉ báo cáo như sensitivity analysis trong phụ lục (§7.6)
- ❌ Gọi hiện tượng ghép fold là **"bias"** — dùng "cross-fold scale confounding" / "aggregation artifact"
- ❌ **Bất kỳ phát biểu "đầu tiên" / "chưa ai làm"** nếu không kèm systematic search được ghi lại

## 12.4 Đóng góp tuyên bố

1. Một **benchmark stress-test cạm bẫy X-quang nha khoa** có nhãn chuyên gia, guideline khóa trước, độ tin cậy đo được.
2. Một **quy trình selective classification không rò rỉ**, đánh giá ở **cả cấp răng và cấp ảnh**, với quy tắc tổng hợp không giả định khả so sánh liên model.
3. Một **cơ chế gating đơn giản, đơn điệu**, so sánh công bằng với các baseline trên cùng backbone.
4. **Phân tích failure mode** của chẩn đoán phân biệt nha khoa dưới nhiễu kim loại và cạm bẫy giải phẫu.

---

# §13. LỘ TRÌNH MÃ NGUỒN & NGÂN SÁCH COMPUTE

## 13.1 Lộ trình

| Bước | File | Nội dung |
|---|---|---|
| 0 | `00_sanity_checks.py` | **§10.1** (phân bố lớp theo fold) và **§10.2** (discriminator hai subset, có điều kiện) — **trước khi cắt patch**. §10.3 và §10.4 **không** chạy ở đây, xem bước 6 |
| 1 | `01_extract_patches.py` | Đọc COCO, loại 3 hộp đa nhãn, iterative stratification → `folds.json`, cắt patch lề 6%, **lưu kích thước & diện tích gốc** |
| 2 | `02_dataset_loader.py` | PyTorch Dataset, augmentation **khóa** §4.2 (CLAHE/flip là cờ **chỉ dùng cho ablation §9**) |
| 3 | `03_train_backbone.py` | ResNet-50 CE, LR 1e-4, AMP, **1 model/vòng CV**, **ghi wall-clock vòng 1** |
| 4 | `04_compute_manifold.py` | PCA (train folds) với **`d = 64` cố định** → μ_k, Σ_shrunk, **và μ₀, Σ₀ cho RMD** |
| 5 | `05_fit_gate.py` | Φ_S, Φ_M, α\*, T\*, τ trên validation fold của từng vòng (`d` đã cố định ở bước 4) |
| 6 | `06_evaluate.py` | **PRIMARY:** C-AURC **trong từng fold** (§7.1) → `AURC_CV` macro-average (§7.2); E-AURC; Err-AUROC; OracleCov + ValCalibrated per-fold → macro-average; cấp răng + cấp ảnh; class-balanced. **Chạy assertion §5.5 trong TỪNG fold.** **SECONDARY:** nhánh §7.6 — chuẩn hóa ECDF theo fold cho **mọi** phương pháp rồi ghép. **DIAGNOSTIC:** §10.3 (hồi quy `g(x)` ~ diện tích crop) và §10.4 (đếm unique score per-fold) |
| 7 | `07_statistics.py` | **Bootstrap lồng §8.1** (resample ảnh trong từng fold — `T_f` cho cấp răng, `I_f` cho cấp ảnh; giữ cấu trúc fold, ghép cặp), CI cho `AURC_CV` và `ΔAURC_CV`, Holm, ước lượng ICC |
| 8 | `08_figures.py` | RC curves (hai cấp, per-fold + macro-average), scatter (Φ_S,Φ_M), t-SNE, Grad-CAM |

> **Ràng buộc cài đặt:** trong toàn bộ codebase **không được tồn tại hàm nào tính AURC bằng cách sort điểm thô của cả 5 fold chung một bảng.** Chỉ có hai đường: per-fold (§7.1) và pooled-normalized (§7.6).

## 13.2 Ngân sách compute — ƯỚC TÍNH, chưa xác nhận

| Hạng mục | Số lần train |
|---|---|
| **Thí nghiệm chính (5-fold)** | **5** |
| Deep Ensembles (vòng 1, K=5) — supplementary | +4 |
| Ablation (CLAHE, flip, Center Loss) — thứ cấp | +3 |
| **TỔNG** | **12** |

Các phương pháp hậu nghiệm và mọi tham số §5.3-B: **0 lần train thêm**.

> **⚠️ Thời gian là ƯỚC TÍNH CHƯA KIỂM CHỨNG.** Đo **wall-clock thực của vòng CV 1**, ngoại suy tuyến tính cho 11 lần train còn lại, ghi vào §14. Trước khi có nó, **không cam kết thời gian với ai**.
> Colab free có giới hạn phiên và GPU không cố định → checkpoint sau mỗi vòng.

---

# §14. DEVIATION LOG

| Ngày | Mục | Thay đổi | Lý do | Ảnh hưởng claim? |
|---|---|---|---|---|
| 2026-09-29 | §2.1, §7.6 | (1) "3 hộp mang 2 nhãn" thực chất là 3 vị trí hộp có >1 annotation — `image_id=447` là Caries lặp hai lần, không phải hai nhãn khác nhau; vẫn loại cả 3 → 3.523 patch không đổi. (2) Số mẫu cấp patch sau khi loại: Impacted 604 / Caries 2.186 / Periapical 157 / Deep Caries 576 (bộ 604 / 2.189 / 158 / 578 là cấp annotation) | Sửa số liệu mô tả: §2.1 đếm ở cấp annotation, bổ sung cột cấp patch; §7.6 n=158 → 157 | Không |
| 2026-09-29 | §3.1, §4.1 | Chi tiết cài đặt trong `01_extract_patches.py`, ghi **trước khi** sinh `folds.json` thật: (1) `IterativeStratification(n_splits=5, order=1)` cài bằng `iterstrat.MultilabelStratifiedKFold(shuffle=True, random_state=42)` — gói `iterative-stratification` đã liệt kê ở §3.5, cùng thuật toán Sechidis 2011 bậc 1; seed 42 là hằng số trong code, không có tham số dòng lệnh để đổi. (2) Ma trận nhãn 705×4 tính trên 3.523 patch giữ lại; đã kiểm tra: trùng khớp hoàn toàn với ma trận tính trên 3.529 annotation. 27 ảnh không có patch nào (hàng toàn 0) vẫn được chia vào fold. (3) Crop §4.1 cài bằng cách mở rộng từng cạnh 6% rồi cắt theo biên ảnh (công thức `w_new = min(W, 1.12w)` trong §4.1 có thể vượt biên phải khi cạnh trái bị cắt); tọa độ float làm tròn ra ngoài (floor trái/trên, ceil phải/dưới). Với dữ liệu này có 0 hộp chạm biên, nên hai cách cho cùng một crop, chỉ khác phần làm tròn nguyên | Plan không khóa thư viện và cách làm tròn; công thức §4.1 viết tắt | Không |
| 2026-09-29 | §3.1 / §3.5 | §3.1 viết `IterativeStratification(n_splits=5, order=1)`, là cú pháp skmultilearn; trong package `iterative-stratification` mà §3.5 liệt kê, `IterativeStratification` là một hàm `(labels, r, random_state)`, không phải splitter class. Cài đặt dùng `iterstrat.MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, mà bên trong gọi chính hàm đó với `r=[1/5]*5` — Sechidis bậc một, đúng package §3.5 | Plan tự mâu thuẫn giữa API và package; hai cài đặt cùng thuật toán cho phân chia fold khác nhau nên phải ghi rõ cái nào sinh ra `folds.json` | Không |
| 2026-09-29 | §7.3 | Phát hiện: 27 trong 705 ảnh **không có annotation nào ngay từ đầu** (image_id 3, 18, 36, 100, 173, 206, …), không phải do loại 3 hộp đa nhãn; 678 ảnh có ≥1 annotation. Với các ảnh này, điều kiện "MỌI răng có nhãn đều có score ≥ τ" đúng một cách rỗng → ảnh luôn được nhận ở mọi τ và không bao giờ chứa răng sai, đẩy image-level coverage lên và image-level risk xuống cho mọi phương pháp. Quyết định: loại 27 ảnh này khỏi phân tích cấp ảnh; mẫu số §7.3 là số ảnh có ≥1 răng mang nhãn trong fold, tổng n = 678. Phân tích cấp răng không đổi; `folds.json` không đổi (27 ảnh vẫn nằm trong fold) | Định nghĩa §7.3 cũ cho giá trị rỗng-đúng không mang thông tin về gate | **Có** — image-level coverage/risk khác so với khi tính trên 705 |
| 2026-09-29 | §8.1 | Pseudo-code bootstrap chỉ viết "resample các ẢNH trong test fold f", không nêu tập ảnh nào; CI của metric cấp ảnh vì thế có thể bốc lại 27 ảnh không annotation và tái lập thiên lệch rỗng-đúng mà §7.3 (dòng trên) đã loại. Quyết định, khóa theo họ metric: (i) **cấp ảnh** — resample trong `I_f` (ảnh có ≥1 răng mang nhãn), bắt buộc vì metric định nghĩa trên `I_f`; (ii) **cấp răng** — resample từ `T_f` = toàn bộ ảnh trong test fold, kể cả ảnh rỗng, để giữ biến thiên "có phim không mang tổn thương nào" (ảnh rỗng đóng góp 0 patch). Hai lần bốc độc lập trong cùng lần lặp `b`; ghép cặp giữa các phương pháp giữ trong từng họ. Sửa kèm: "số đơn vị độc lập gần với 705 ảnh" → 678 | Điểm ước lượng cấp ảnh tính trên 678 ảnh mà CI tính trên 705 là hai tổng thể khác nhau; cấp răng có hai cách đều bảo vệ được nên phải khóa một cách trước khi viết code | **Có** — CI của metric cấp ảnh |

**Benchmark compute thực đo (điền sau vòng CV 1):**

| Hạng mục | Giá trị đo được |
|---|---|
| GPU thực tế | |
| Thời gian train vòng 1 | |
| Ngoại suy tổng 12 lần train | |

**Kết quả assertion §5.5 — điền cho TỪNG fold:**

| Fold | Số cặp nghịch thế (phải = 0) | Số ties mới do Φ_S | \|AURC_f(R₁) − AURC_f(MSP)\| |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

**Sanity check nhánh secondary (§7.6):**

| Kiểm tra | Giá trị |
|---|---|
| \|C-AURC_pooled-norm(α=1) − C-AURC_pooled-norm(MSP)\| | |

**Tham số hậu nghiệm đã chọn:**

| Tham số | Giá trị |
|---|---|
| `d` | **64 — khóa tiên nghiệm, không điền từ dữ liệu** |
| `α*` fold 1..5 | |
| `T*` fold 1..5 | |

---

# §15. CHANGELOG v5 → v6

**Phạm vi: chỉ quy tắc tổng hợp và cách gọi tên.** Không đổi câu hỏi nghiên cứu, dataset, kiến trúc, số lần huấn luyện, hay bất kỳ thí nghiệm nào.

| # | Mục | Thay đổi |
|---|---|---|
| 1 | §0 | Tách **tập dữ liệu** khỏi **đơn vị đánh giá**. "OOF evaluation cohort" → "**OOF prediction set**", kèm cảnh báo rằng nó không phải đơn vị tính metric |
| 2 | §3.2 | Viết lại lý do dùng 5-fold: mỗi ảnh đóng góp một lần; `AURC_CV` trung bình 5 lần train; nhánh secondary giữ cỡ mẫu cho lớp hiếm/subgroup. Bỏ tuyên bố "primary cần toàn bộ 705 ảnh như một tập duy nhất" |
| 3 | §3.4 | `OracleCov-Risk` tính **per-fold → macro-average**. `ValCalibrated-τ` giữ nguyên cơ chế, thêm quy định báo cáo ba phần (risk macro-average, coverage thực đạt macro-average, bảng 5 giá trị per-fold) |
| 4 | §5.3-A/B | **`d` chuyển sang bảng A và khóa tiên nghiệm = 64**, không chọn từ dữ liệu. Phương án "trung bình AURC(val) qua 5 vòng" bị loại vì **gây rò rỉ**: validation fold của vòng 2 là F1, chính là test fold của vòng 1. Căn cứ quy ước: N/d ≈ 33 ở d=64 (so với ≈16 ở 128, ≈8 ở 256). Độ nhạy theo `d` vẫn báo cáo đủ ở §9.5. `α`, `T`, `τ` vẫn per-fold — không rò rỉ vì val fold của một vòng không bao giờ là test fold của chính vòng ấy |
| 5 | §5.5 | Assertion chuyển vào **trong từng fold**: `AURC_f(R_1) = AURC_f(MSP)`, f = 1..5. **Bỏ** mọi yêu cầu về pooled ranking. Bảng ghi kết quả §14 mở rộng thành 5 dòng |
| 6 | §7.0 (mới) | Thêm mục giải thích **vì sao không ghép điểm thô**: điều kiện khả so sánh liên model, các nguồn vi phạm (thang logit, early stopping khác epoch, Σ/PCA/β\* khác), và mức nhạy khác nhau theo phương pháp |
| 7 | §7.1 | C-AURC định nghĩa **trong phạm vi một test fold** |
| 8 | §7.2 (mới) | **`AURC_CV = (1/5)·Σ AURC_f` là PRIMARY ENDPOINT.** Thêm quy tắc **macro-average không trọng số cho metric dạng tỷ số**, kèm bắt buộc báo cáo 5 giá trị per-fold |
| 9 | §7.3 | Image-level coverage/risk tính **per-fold → macro-average** |
| 10 | §7.4 | Thêm cảnh báo cỡ mẫu: Periapical per-fold ≈ 32; phân tích lớp hiếm có ý nghĩa nằm ở nhánh secondary |
| 11 | §7.5 | ΔAURC tính trên **`AURC_CV`**. **AUROC ở decile tự tin cao chuyển sang nhánh SECONDARY** (per-fold chỉ ≈70 patch) — vị trí khóa trước |
| 12 | §7.6 (mới) | Thêm **fold-wise rank-normalized pooled analysis** làm nhánh secondary/sensitivity: định nghĩa 4 bước, áp **đồng nhất cho mọi phương pháp kể cả đề xuất**, tên gọi bắt buộc, phát biểu chuẩn về bản chất, danh sách phân tích được phục vụ, sanity check phụ, và **cấm dùng raw-vs-normalized gap làm quy tắc quyết định** |
| 13 | §8.1 | **Bootstrap lồng**: resample ảnh trong từng fold, giữ cấu trúc fold, giữ paired predictions, không resample patch, không resample fold. Pseudo-code đầy đủ |
| 14 | §8.2 | So sánh chính thực hiện trên **`ΔAURC_CV`** |
| 15 | §8.3 | Nêu rõ CI phải đến từ bootstrap §8.1, không từ 5 giá trị `AURC_f` |
| 16 | §8.4 | Viết lại: `AURC_CV` là trung bình của 5 AURC, mỗi cái thuộc đúng một mô hình, **không giả định khả so sánh liên model** — mục này mạnh lên. Nêu rõ nhánh secondary ước lượng một đại lượng khác, không thay thế |
| 17 | §11.1 | Subgroup: primary per-fold → macro-average; **khai báo trước** rằng per-fold có thể không kết luận được, kết quả thực tế nằm ở nhánh secondary, Limitations phải ghi nhận. **Không ước đoán con số prevalence** |
| 18 | §12.3 | Thêm 4 mục cấm: pooled raw AURC làm primary; mô tả normalization như cải tiến baseline; dùng raw-vs-normalized gap làm quy tắc quyết định; gọi hiện tượng ghép là "bias" |
| 19 | §13.1 | Cập nhật mô tả `06_evaluate.py` và `07_statistics.py`. Thêm ràng buộc: **không tồn tại hàm nào tính AURC bằng cách sort điểm thô của cả 5 fold** |
| 20 | §14 | Bảng assertion mở rộng thành 5 fold; thêm bảng sanity check nhánh secondary; thêm bảng ghi `d`, `α*`, `T*` |
| 21 | §16 | Thêm 4 mục checklist trước khi code, 3 mục checklist trước bản thảo |
| 22 | Toàn tài liệu | Từ **"bias"** (mô tả hiện tượng ghép fold) → **"cross-fold scale confounding" / "aggregation artifact"**. Phát biểu về ECDF viết lại chính xác: *"does not change the within-fold ranking or within-fold AURC, but it changes the cross-fold aggregation rule when scores are pooled"* |
| **23** | §7.1 bước 6–7 | **Sửa lỗi miền tích phân.** v5 tích phân trên `[c_1, 1]`, nhưng `c_1` phụ thuộc kích thước tie block đầu tiên nên **khác nhau giữa các phương pháp** → AURC tính trên miền khác nhau, không so sánh được. v6 mở rộng hằng số `r(c) = r_1` trên `(0, c_1]` và tích phân trên **`[0, 1]` cho mọi phương pháp** |
| **24** | §8.1 | Thêm quy tắc xử lý **lần lặp bootstrap suy biến** cho nhóm con nhỏ: bỏ lần lặp và đếm; nếu >5% thì không báo cáo CI cho nhóm đó. Tránh hành vi không xác định và tránh thay bằng giá trị mặc định |
| **25** | §10 (đầu mục), §13.1 | **Sửa lỗi thời điểm chạy.** v5 xếp cả 4 kiểm tra vào `00_sanity_checks.py` "trước khi cắt patch" — **không thực hiện được**: §10.3 cần `g(x)` (phải có backbone + manifold) và §10.4 cần điểm số mọi phương pháp. Tách rõ: §10.1–10.2 là **chặn đường** trong `00_sanity_checks.py`; §10.3–10.4 là **diagnostic báo cáo** trong `06_evaluate.py`. §10.2 đánh dấu **có điều kiện** (chỉ cần nếu chạy §12.1 hoặc dùng pilot phương án (a)) |
| **26** | §9 | **Tách phạm vi fold theo chi phí.** Ablation không cần train thêm (9.1, 9.5, 9.6) chạy **cả 5 fold** và báo cáo `AURC_CV` — chi phí gần 0 vì chỉ tính lại từ feature đã lưu. Chỉ ablation cần train (9.2, 9.3, 9.4) giữ ở vòng CV 1, báo cáo `AURC_1` kèm nhãn "single CV round". Nêu rõ **§9.1 chính là chẩn đoán §7.5 #1**, trình bày một lần |
| **27** | §0, §7.1 bước 8, §7.2 tiêu đề, §9.5 | Sửa nhỏ: định nghĩa **clinical annotation cohort** thành có điều kiện theo phương án pilot (a)/(b); bước 8 nêu rõ `c* < c_1 ⇒ r(c*) = r_1`; đổi tiêu đề "metric dạng tỷ số" → "mọi metric tính trong fold" cho khớp nội dung; sửa lỗi in đậm sai giá trị trong lưới `d` |
| **28** | §7.4.1 (mới) | **Phân tích 3-class được định nghĩa dứt khoát: LỌC HẬU NGHIỆM**, không huấn luyện lại — ngân sách 12 lần train không đổi. Nêu rõ hai ràng buộc: lọc theo **nhãn thật** (không theo nhãn dự đoán, vì đó là điều kiện hóa trên kết quả), và mẫu bị đoán nhầm **thành** Impacted **vẫn tính là lỗi**. Ghi rõ phân tích này trả lời câu gì và **không** trả lời câu gì |
| **29** | §7.4.2 (mới) | **Class-balanced selective risk được định nghĩa bằng công thức**, thay vì chỉ nêu lý do. Chọn dạng trọng số mẫu `w_i = 1/(K·n_{y_i,f})`: rút gọn đúng về balanced error rate khi nhận toàn bộ, không bao giờ chia 0 theo lớp, và chỉ thay **một** thành phần trong C-AURC nên dùng lại được cùng hàm. **Coverage vẫn không trọng số** vì nó là đại lượng vận hành |
| **30** | §7.5 #2 | **Định nghĩa đầy đủ chẩn đoán decile**: decile lấy theo **Φ_S đã chuẩn hóa** (không phải Φ_M, không phải R_α — hai cái sau sẽ điều kiện hóa trên chính tín hiệu đang đánh giá), AUROC của Φ_M trong nhóm đó, báo cáo kèm số ca sai và CI. Cấm hạ ngưỡng decile để lấy thêm mẫu |

---

# §16. FINAL LOCK CHECKLIST

## A. Trước khi viết dòng code đầu tiên

**Rò rỉ và hiệu chỉnh**
- [ ] Đối chiếu bảng §3.3 với từng dòng code, không ngoại lệ
- [ ] PCA / μ_k / Σ_shrunk / μ₀ / Σ₀ / không gian con ViM: **chỉ train folds**
- [ ] Φ, α, T, τ: **chỉ validation fold**

**Cấu hình huấn luyện**
- [ ] Tham số §5.3-A khóa trong code dưới dạng hằng số, không cờ tìm kiếm
- [ ] **Không vòng lặp nào quét LR, CLAHE hay flip trong `03_train_backbone.py`** — đúng 5 lần train
- [ ] `folds.json` tất định, đã commit, không nhánh code nào sinh lại

**Quy tắc tổng hợp — mới ở v6**
- [ ] **Không tồn tại hàm nào tính AURC bằng cách sort điểm THÔ của cả 5 fold chung một bảng**
- [ ] **`d = 64` là hằng số trong code**, không có nhánh nào chọn `d` từ dữ liệu ngoài ablation §9.5
- [ ] **AURC tích phân trên `[0,1]`** với mở rộng hằng số trên `(0, c_1]` — cùng miền cho mọi phương pháp
- [ ] **Bootstrap giữ cấu trúc fold**, resample ở cấp ảnh — **từ `T_f` (toàn bộ ảnh fold) cho metric cấp răng, từ `I_f` (ảnh có ≥1 răng mang nhãn) cho metric cấp ảnh** (§8.1) — ghép cặp giữa các phương pháp trong từng họ metric, có xử lý lần lặp suy biến
- [ ] **Nhánh §7.6 áp chuẩn hóa ECDF đồng nhất cho MỌI phương pháp**, kể cả phương pháp đề xuất

**Công bằng và metric**
- [ ] Một backbone CE thuần cho mọi phương pháp §6.1; MC-Dropout đã loại kèm lý do; Deep Ensembles ở bảng phụ
- [ ] C-AURC §7.1 là hàm **duy nhất** tính AURC trong codebase
- [ ] Coverage/risk cấp răng và cấp ảnh đều tính per-fold rồi macro-average; **mẫu số cấp ảnh là `|I_f|`, không phải số ảnh trong fold** (§7.3)
- [ ] Macro-average **không trọng số** cài đặt nhất quán cho mọi metric tỷ số
- [ ] Danh sách tham số §5.3 đã đóng · So sánh chính §8.2 đã khai báo
- [ ] **3-class cài đặt là lọc hậu nghiệm theo NHÃN THẬT** (§7.4.1), không có nhánh train model 3 lớp
- [ ] **Class-balanced risk dùng đúng trọng số `w_i = 1/(K·n_{y_i,f})`** (§7.4.2); coverage không trọng số
- [ ] **Decile ở chẩn đoán §7.5 #2 lấy theo Φ_S đã chuẩn hóa**, không phải Φ_M hay R_α
- [ ] Seed cố định, phiên bản thư viện đã ghi

## B. Trước khi bắt đầu gán nhãn lâm sàng

- [ ] Guideline v1 đã viết
- [ ] **Nguồn 15 patch pilot đã quyết định theo §11.2**, ưu tiên (a); nếu (b) thì danh sách đã ghi và cơ chế loại trừ đã cài trong code
- [ ] Guideline đã **KHÓA** sau pilot
- [ ] Quy trình mù ba lớp đã thiết lập, **gồm mù với output model**
- [ ] Lịch đọc lại ≥3 tuần đã đặt

## C. Trong khi chạy thí nghiệm

- [ ] **§10.1 và §10.2 đã chạy TRƯỚC khi cắt patch**; §10.3 và §10.4 chạy trong `06_evaluate.py`, không nhầm thời điểm
- [ ] **Assertion §5.5 đã chạy trong CẢ 5 FOLD:** 0 cặp nghịch thế mỗi fold; số ties mới và độ lệch AURC ghi vào §14
- [ ] Sanity check nhánh secondary (§7.6) đã chạy và ghi vào §14
- [ ] `α*` (5 giá trị), `T*` (5 giá trị) đã ghi vào §14 (`d` đã khóa = 64, không điền từ dữ liệu)
- [ ] Wall-clock vòng CV 1 đã đo và ghi vào §14
- [ ] Mọi sai lệch đã ghi DEVIATION LOG **trước khi** áp dụng

## D. Trước khi viết bản thảo

**Mô tả thiết kế**
- [ ] Official split vs internal CV phân biệt rõ, kèm câu "không so sánh được với challenge leaderboard"
- [ ] Mọi phát biểu về nguồn gốc dataset đã đối chiếu tài liệu gốc DENTEX và có trích dẫn
- [ ] Toàn bộ văn bản dùng "OOF prediction set", không có "test set" chỉ kết quả cuối

**Quy tắc tổng hợp — mới ở v6**
- [ ] **Primary endpoint ghi rõ là `AURC_CV`** (macro-average của 5 AURC per-fold)
- [ ] **Bảng secondary ghi đúng tên "fold-wise rank-normalized pooled analysis"**, đặt ở mục sensitivity, không phải primary
- [ ] **Quy tắc macro-average không trọng số cho metric tỷ số đã nêu trong Methods**, kèm bảng 5 giá trị per-fold trong phụ lục
- [ ] Phát biểu chuẩn về ECDF (§7.6) xuất hiện đúng nguyên văn khi mô tả nhánh secondary
- [ ] Không chỗ nào gọi fold-wise normalization là cải tiến baseline
- [ ] Không chỗ nào dùng từ "bias" cho hiện tượng ghép fold

**Báo cáo số liệu**
- [ ] Mọi con số có CI từ bootstrap lồng §8.1
- [ ] Risk@coverage có cả hai phiên bản, ghi nhãn oracle rõ; ValCalibrated kèm coverage thực đạt
- [ ] Coverage cấp ảnh báo cáo song song cấp răng, **không có phép tính nhị thức nào**; ghi rõ n = 678 ảnh có nhãn (27 ảnh không annotation bị loại khỏi cấp ảnh, §7.3)
- [ ] Deep Ensembles ở bảng phụ, ghi rõ "single CV round, 5× training cost, not directly comparable"
- [ ] **Ablation §9.2–9.4 ghi rõ "single CV round"**; §9.1, §9.5, §9.6 báo cáo `AURC_CV` trên cả 5 fold
- [ ] Subgroup ghi rõ là subgroup lồng trong OOF prediction set, có CI; nếu CI quá rộng thì nói thẳng là không kết luận được

**Thuật ngữ và giới hạn**
- [ ] Không chỗ nào dùng "OOD", "healthy", "confirmed disease", "single-center"
- [ ] §10.2 diễn giải đúng: không tuyên bố hai phân phối tương đương
- [ ] Tuyên bố độ tin cậy dùng đúng từ (intra- vs inter-rater)
- [ ] Đại lượng ước lượng mô tả đúng bản chất (§8.4)
- [ ] Không có phát biểu "đầu tiên"/"chưa ai làm" thiếu systematic search
- [ ] Danh sách §12.3 đã đối chiếu toàn bản thảo, từng dòng
- [ ] DEVIATION LOG đưa vào phụ lục bài báo
- [ ] Limitations có: một dataset công khai duy nhất; nguồn gốc thu thập chưa kiểm chứng độc lập; không external validation; **phương sai huấn luyện không đo (chỉ 5 lần train, mỗi fold một lần)**; MC-Dropout vắng mặt kèm lý do; cấu hình augmentation khóa trước chứ không tối ưu; **subgroup per-fold có thể không đủ power**

---

## THAM CHIẾU CHÍNH

- Chow (1970) — *On optimum recognition error and reject tradeoff*
- Hendrycks & Gimpel (2017) — MSP
- Gal & Ghahramani (2016) — MC-Dropout *(Related Work, không dùng làm baseline — §6.2)*
- Guo et al. (2017) — Temperature Scaling
- Lakshminarayanan et al. (2017) — Deep Ensembles
- Lee et al. (2018) — Mahalanobis OOD
- Geifman & El-Yaniv (2017, 2019) — Selective classification
- Liu et al. (2020) — Energy-based OOD
- Ren et al. (2021) — Relative Mahalanobis
- Wang et al. (2022) — ViM
- Ledoit & Wolf (2004) — Shrinkage covariance
- Sechidis, Tsoumakas & Vlahavas (2011) — Iterative stratification đa nhãn
- Dietterich (1998) — Cảnh báo kiểm định trên CV
- Hamamci et al. (2023) — DENTEX Challenge
