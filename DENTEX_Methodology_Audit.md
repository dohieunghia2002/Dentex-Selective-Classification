# METHODOLOGY AUDIT — DENTEX SELECTIVE CLASSIFICATION
### Phản biện độc lập đối với phản hồi A/B/C/D về `DENTEX_EMD_Gate_Research_Plan.md`

**Ngày:** 2026-09-28
**Phạm vi:** Audit lại chính các đề xuất A/B/C/D đã đưa ra ở lượt trước, không mặc định chúng đúng.
**Nguyên tắc ràng buộc:** Không đổi research question. Không thêm dataset để nâng tier tạp chí. Mọi đề xuất làm phình scope phải bị đánh dấu 🚩 SCOPE CREEP.

---

## TÓM TẮT KẾT QUẢ AUDIT

Trong 11 đề xuất được audit:

- **4 đề xuất SAI hoặc nói quá mạnh** → tự bác bỏ
- **3 đề xuất ĐÚNG nhưng lý do đưa ra là sai** → giữ, sửa lý do
- **4 đề xuất GIỮ NGUYÊN**
- **1 lỗi leakage nghiêm trọng do chính phản hồi cũ tạo ra** (Deep Ensemble)
- **5 vấn đề mới** chưa được phát hiện ở cả hai lượt trước

---

# PHẦN I — AUDIT ĐỀ XUẤT A: 5-FOLD CROSS-VALIDATION

## 1. Vấn đề định giải quyết có thật không?

Có, nhưng bản chất bị mô tả sai.

**Số liệu đúng:** 3.526 hộp / 705 ảnh ≈ 5,0 patch/ảnh. Test 15% = 106 ảnh ≈ 530 patch, Periapical ≈ 24 mẫu. Các con số này là thật.

**Lỗi trong lập luận cũ:** phát biểu *"8 phương pháp trên 530 mẫu sẽ không phân biệt được"* đã bỏ qua một điều quyết định — toàn bộ 8 phương pháp đều là **hậu nghiệm trên cùng một backbone, chấm điểm cùng một tập dự đoán**. Đây là thiết kế **ghép cặp (paired)**. Phương sai của hiệu số ΔAURC nhỏ hơn rất nhiều so với phương sai của từng AURC riêng lẻ, vì thành phần "tập test này khó hay dễ" triệt tiêu khi lấy hiệu. Trên 530 patch ghép cặp, CI của ΔAURC hoàn toàn có thể đủ hẹp để kết luận.

> **TỰ BÁC BỎ:** phát biểu "kết quả sẽ không kết luận được gì" là nói quá.

**Cái vẫn đúng:** phân tích theo lớp và theo nhóm nhỏ thì vô vọng. Periapical n≈24 trong test; artifact subgroup có lẽ 50–80 patch. Mọi kết luận kiểu *"gate giúp nhiều nhất ở ca có vệt kim loại"* sẽ không có power. **Đó mới là lý do thật để cần 5-fold.**

## 2. OOF predictions ≠ tăng power tuyến tính

| Khía cạnh | Kết luận |
|---|---|
| **Số OOF prediction** | 3.523 (sau khi loại 3 hộp đa nhãn) |
| **Số bệnh nhân độc lập** | Vẫn là **705** — không tăng một đơn vị nào |
| **Nguồn power thêm** | Mỗi bệnh nhân được dùng làm test đúng 1 lần, thay vì chỉ 15% bệnh nhân |

### 2.1 Tương quan trong ảnh (clustering)

Các patch cùng một phim chia sẻ máy chụp, liều tia, giải phẫu bệnh nhân, mức nhiễu kim loại. ICC chắc chắn dương và không nhỏ.

Với m = 5 patch/ảnh và ICC = 0,2:

```
design effect = 1 + (m − 1) × ICC = 1 + 4 × 0,2 = 1,8
n_hiệu_dụng  = 3.523 / 1,8 ≈ 1.960   (không phải 3.523)
```

Tỷ lệ tăng so với single split vẫn giữ (~6,6×) vì cả hai cùng chịu hệ số đó.

> **Hệ quả bắt buộc:** bootstrap phải ở **cấp ảnh (cluster bootstrap)**, không phải cấp patch. Bootstrap cấp patch cho CI hẹp giả tạo — đây là lỗi thống kê phổ biến nhất trong loại bài này.

### 2.2 Phương sai AURC

Hai nguồn phương sai:

1. **Nguồn lấy mẫu** (bệnh nhân nào rơi vào test) — cluster bootstrap bắt được.
2. **Nguồn huấn luyện** (seed, khởi tạo) — cluster bootstrap **không** bắt được.

Phải khai báo rõ giới hạn này trong bài.

### 2.3 Suy luận thống kê từ OOF

AURC tính trên OOF gộp **không phải AURC của một mô hình**. Nó là AURC của *quy trình huấn luyện*, ước lượng qua hỗn hợp 5 mô hình khác nhau. Đây là đại lượng ước lượng hợp lệ — và thậm chí phù hợp hơn với câu hỏi "phương pháp gating này có hoạt động không" — nhưng **phải gọi đúng tên trong bài**.

> **CẤM:** dùng 5 giá trị AURC của 5 fold làm 5 quan sát độc lập để chạy t-test. Các tập train chồng lấn 75%, giả định độc lập sai hoàn toàn (Dietterich 1998). Nếu muốn đi hướng đó phải dùng hiệu chỉnh Nadeau–Bengio.

## 3. Protocol trong từng fold

5-fold thuần chỉ cho train/test, **không có val**. Nhưng val là bắt buộc để fit Φ_S, Φ_M, α, T, τ.

Cấu hình đúng — xoay vòng:

| Fold | Train | Val | Test |
|---|---|---|---|
| 1 | F2, F3, F4 | F5 | F1 |
| 2 | F3, F4, F5 | F1 | F2 |
| 3 | F4, F5, F1 | F2 | F3 |
| 4 | F5, F1, F2 | F3 | F4 |
| 5 | F1, F2, F3 | F4 | F5 |

Gắn từng bước vào đúng một tập:

| Thành phần | Ước lượng trên | Ghi chú |
|---|---|---|
| Backbone | 3 train folds | Không bao giờ thấy val hoặc test |
| μ_k, Σ_shrunk | 3 train folds | Tham số mô hình mật độ, **không** dùng val |
| Phép giảm chiều (PCA) | 3 train folds | Xem Phần III |
| Φ_S, Φ_M | val fold | ECDF thực nghiệm, average rank |
| α | val fold | Quét lưới, tối ưu AURC(val) |
| T (temperature) | val fold | |
| τ | val fold | Xem mục 4 |
| OOF | Ghép test fold từ cả 5 fold | |

## 4. Lỗi leakage mới phát hiện: chọn ngưỡng τ

Mục 4.1.3 của plan yêu cầu báo cáo *"Selective Risk tại Coverage = 70/80/90%"*. Cách làm mặc định là sắp xếp tập test theo điểm rồi cắt tại 80% — tức **ngưỡng τ được chọn bằng chính tập test**. Đó là **oracle coverage**, và nó không phải cái một phòng khám làm được: ngoài đời phải quyết định τ trước khi thấy bệnh nhân.

**Bắt buộc phân biệt hai con số:**

| Phiên bản | Cách tính | Vai trò |
|---|---|---|
| **Risk@oracle-coverage** | Cắt trên test | So sánh với văn liệu. Ghi nhãn rõ |
| **Risk@calibrated-τ** | τ chọn trên val để đạt coverage mục tiêu, áp lên test, **báo cáo coverage thực tế đạt được** | Con số lâm sàng — lấy làm chính |

AURC không dính vấn đề này vì nó không cần ngưỡng.

## 5. Deep Ensemble — LỖI NGHIÊM TRỌNG trong phản hồi cũ

**Claim cũ:** *"Deep Ensembles gần như miễn phí vì đã có sẵn 5 model từ 5 fold."*

### 5.1 Chứng minh claim này sai

- `model_1` được train trên F2, F3, F4.
- Muốn ensemble cho test fold F1, cần nhiều model dự đoán F1.
- Nhưng `model_2` (train trên F3, F4, F5) **đã nhìn thấy F1** trong tập train của nó.
- `model_3`, `model_4`, `model_5` cũng vậy.
- **Chỉ duy nhất `model_1` là sạch với F1.**

Ensemble `model_1..model_5` trên F1 = **đưa dữ liệu huấn luyện vào tập test**.

### 5.2 Vì sao đặc biệt nguy hiểm với đề tài NÀY

Hậu quả không chỉ là accuracy bị thổi phồng. Nghiêm trọng hơn: **mẫu đã được ghi nhớ sẽ có softmax bão hòa gần 1,0** — tức là bơm độ tự tin giả vào đúng đại lượng mà cả bài báo đang đo. Deep Ensembles sẽ trông cực tốt vì lý do hoàn toàn nhân tạo.

### 5.3 Protocol không leakage

Với mỗi fold, train K model **khác seed trên cùng bộ train folds của fold đó**.

**Chi phí thật:**

| Hạng mục | Số lần train | Thời gian ước tính (T4, AMP) |
|---|---|---|
| Post-hoc baselines (MSP, TS, Mahalanobis, RMD, Energy, ViM, đề xuất) | 5 | ~1 giờ |
| MC-Dropout | 0 (chỉ 20 forward pass) | vài phút |
| **Deep Ensembles đầy đủ** | **25** (5 fold × 5 seed) | **~5 giờ** — vượt giới hạn phiên Colab free (~4h) |

**Đề xuất thực tế:** chạy Deep Ensembles **trên fold 1 duy nhất với K=5** (thêm 4 lần train ≈ 50 phút), báo cáo như một điểm tham chiếu một-fold, ghi chú rõ *"5× chi phí huấn luyện"* và *"CI rộng hơn các phương pháp khác"*.

### Kết luận A

**GIỮ 5-fold CV, nhưng vì lý do khác với lý do ban đầu** — không phải vì so sánh chính thiếu power (thiết kế ghép cặp đã lo phần đó), mà vì phân tích theo lớp hiếm và theo nhóm cạm bẫy thì single split không đủ.
**BÁC BỎ** claim ensemble miễn phí.

---

# PHẦN II — AUDIT ĐỀ XUẤT B: UNLABELED TEETH POOL

## 6. Phân biệt thuật ngữ

| Thuật ngữ | Nghĩa | Áp dụng được? |
|---|---|---|
| **Unlabeled** | Không tồn tại annotation. Không khẳng định gì về răng | ✅ Đúng, và là điều duy nhất biết chắc |
| **Healthy** | Khẳng định dương tính rằng không có bệnh lý | ❌ Cần annotation. Không có |
| **Negative** | Nhãn = không thuộc 4 lớp | ❌ Cần annotation. Không có |
| **Unknown** | Trạng thái nhận thức của ta về răng đó | ✅ Chính xác nhất |
| **Near-distribution** | Cùng modality, cùng giải phẫu, không gian nhãn có thể chồng lấn | ✅ Mô tả đúng |
| **OOD** | Sinh từ phân phối tách rời phân phối train trong không gian nhãn | ❌ **Sai.** Một răng không nhãn hoàn toàn có thể đang bị sâu ngà |

> **TỰ BÁC BỎ:** trong phản hồi cũ, câu *"phương pháp tốt = giữ coverage 80% mà vẫn từ chối nhiều răng lạ"* đã **lén biến unlabeled thành negative**. Đó chính xác là lỗi Negative Sampling Fallacy mà plan v1 đã đúng khi cảnh báo, chỉ tái lập ở một tầng khác.

## 7. Abstention rate tại coverage ghim — có phải metric không?

**KHÔNG. Nó là một thống kê mô tả, không phải metric hiệu năng.**

Nó đo đúng một thứ: **độ dịch chuyển của phân phối điểm tin cậy giữa răng-có-nhãn-bệnh và răng-không-nhãn.** Nó không đo tính đúng đắn của bất kỳ quyết định nào, vì không có quyết định nào có đáp án.

### 7.1 Failure case: phương pháp reject 100% pool

**Không tốt, cũng không xấu — vô nghĩa.** Ba khả năng không phân biệt được:

1. Gate thật sự nhạy với răng bất thường.
2. Gate đang phát hiện khác biệt tiền xử lý giữa hai subset.
3. Mọi răng ngoài tập annotated đều nằm ngoài manifold một cách tầm thường, vì manifold chỉ được dựng từ răng bệnh.

Tệ hơn: nếu trong pool có răng đang sâu ngà thật và model nói "Deep Caries" với độ tin cậy cao, **từ chối là quyết định SAI** — nhưng metric lại tính nó là điểm cộng.

### 7.2 Control tối thiểu nếu vẫn dùng

- Đo song song tỷ lệ từ chối trên **răng có nhãn thuộc cùng những ảnh đó**.
- So sánh với một điểm số ngây thơ (độ sắc nét ảnh, kích thước patch). Nếu độ sắc nét cũng từ chối được 90% pool, gate không chứng minh được gì.

## 8. Lối thoát: đổi mục đích chi tiêu thời gian chuyên gia

Pool chỉ hữu dụng khi **có nhãn**. Thay vì dùng toàn bộ thời gian bác sĩ để đánh dấu cạm bẫy, dành một phần để đọc **mẫu ngẫu nhiên ~300 răng từ pool** với ba lựa chọn:

- không thấy bệnh lý thuộc 4 lớp
- có bệnh lý (ghi lớp)
- không xác định được

Khi đó pool biến từ mô tả thành **tập test thật có lớp thứ 5 (none-of-the-four)**, và tính được risk thực:
- răng lớp 5 → hành động đúng là **abstain**
- răng có bệnh → hành động đúng là **phân loại đúng**

Chi phí ước tính: **3–5 giờ đọc**. Đây là cách dùng thời gian chuyên gia có giá trị cao nhất trong cả đề tài.

## 9. Kiểm tra overlap 705 vs 634

**Bắt buộc?** Có — vì rẻ (một phép giao tập tên file) và diễn giải toàn bộ stress-test phụ thuộc vào nó.

**"Đòn chí mạng" nếu không overlap?** *Nói quá.* Hai subset cùng thuộc một challenge, nhiều khả năng cùng cơ sở và cùng máy; khác biệt nằm ở quy trình gán nhãn chứ chưa chắc ở ảnh. Đây là **mối đe dọa tính hiệu lực cần kiểm chứng**, không phải confound đã chứng minh.

### 9.1 Kiểm chứng cụ thể

Train một phân loại nhị phân đơn giản (logistic regression trên feature, hoặc CNN nhỏ) để phân biệt ảnh thuộc subset A hay subset B:

| Kết quả | Diễn giải |
|---|---|
| AUC ≈ 0,5 | Không có dịch chuyển phát hiện được → pool dùng được |
| AUC ≈ 0,9+ | Có dịch chuyển miền → mọi kết quả trên pool phải khai báo là **cross-subset distribution shift**, tuyệt đối không gọi là OOD |

### 9.2 Chi phí ẩn của phương án "same-image unlabeled teeth"

Trong 705 ảnh disease, mỗi phim có ~28–32 răng nhưng chỉ ~5 răng được gán hộp. **Không có bounding box cho những răng còn lại.** Muốn cắt chúng phải train một tooth detector trên `quadrant_enumeration` rồi suy luận trên 705 ảnh disease — tức thêm nguyên một giai đoạn phát hiện đối tượng mà plan đã cố ý tránh khi chọn Hướng 2.

> 🚩 **SCOPE CREEP — TỰ BÁC BỎ.** Phản hồi cũ đề xuất phương án này mà không nêu chi phí.

### Kết luận B

**ĐƯA POOL KHÔNG NHÃN RA KHỎI THÍ NGHIỆM CHÍNH.**

Stress-test chính chạy trên **răng CÓ nhãn bệnh nhưng kèm cạm bẫy** (metallic artifact / cervical burnout) — thứ plan đã có sẵn, không cần hạ tầng mới. Nó trả lời trọn vẹn câu hỏi của tiêu đề: sai sót tự tin thái quá *trong chẩn đoán phân biệt* khi có nhiễu.

Pool không nhãn chỉ vào bài như **thí nghiệm phụ, mô tả**, và chỉ khi có nhãn chuyên gia theo mục 8.

**Hệ quả cho tiêu đề:** cụm "Under Anatomical and Metallic Artifacts" vẫn hơi gợi ý false-positive trên răng lành. Sửa một cụm là đủ, không đổi research question — thêm **"in Differential Diagnosis"** hoặc **"among Pathological Teeth"**.

---

# PHẦN III — AUDIT ĐỀ XUẤT C

## 10. d = 128 → d = 64: lý do đưa ra là SAI

### 10.1 N = 158 là gì?

**Toàn dataset**, không phải mỗi fold. Trong một fold, train chiếm 3/5 → Periapical ≈ **95 mẫu** (còn ít hơn con số 126 đã nêu).

### 10.2 Điều kiện N_k > d có bắt buộc không? **KHÔNG**

Đây là chỗ sai về mặt khái niệm trong phản hồi cũ:

- Với **tied covariance**, Σ được ước lượng từ **toàn bộ mẫu gộp của mọi lớp** (≈2.100 trong 3 fold), không phải từ 95 mẫu của lớp hiếm. Điều kiện hạng liên quan đến **N_tổng**, không phải N_k.
- **μ_k chỉ là một trung bình mẫu** — không cần điều kiện hạng nào. Sai số của nó co theo √(tr(Σ)/N_k): đó là vấn đề *độ chính xác*, không phải *tính hợp lệ*.
- **Ledoit-Wolf làm Σ luôn khả nghịch** bất kể N.

> **TỰ BÁC BỎ:** phát biểu "vi phạm ràng buộc N_k > d của chính bạn" là sai.

### 10.3 Cái có thật

μ của lớp Periapical nhiễu hơn hẳn ba lớp còn lại → D² tới tâm đó bị thổi phồng có hệ thống → gate có thể thiên vị từ chối ca periapical. Đây là **giả thuyết thực nghiệm kiểm được**, không phải lỗi thiết kế.

### 10.4 d = 64 có tốt hơn không?

Không có cơ sở tiên nghiệm. Nó là lựa chọn kỹ thuật bảo thủ. **Không được đưa vào bài như một quyết định có nguyên tắc.**

**Cách làm đúng:** coi d là siêu tham số, quét lưới **{32, 64, 128, 256}** chọn theo AURC trên **val fold**, báo cáo độ nhạy theo d trong phụ lục. Chọn siêu tham số trên val là hợp lệ, không phải leakage.

### 10.5 Vấn đề quan trọng hơn con số d: CÁCH giảm chiều

Plan viết *"tầng chiếu tuyến tính hoặc Adaptive Average Pooling"*.

- Nếu dùng **tầng chiếu học được trong backbone** → đã thay đổi backbone → mọi baseline (ViM, Energy, MSP) chạy trên feature khác → **phá vỡ tuyên bố công bằng ở mục 4.2**.
- Cách sạch: **PCA khớp trên train folds, áp dụng hậu nghiệm**, không đụng backbone. ViM vốn cũng dùng một không gian con chính → PCA đặt hai phương pháp lên cùng nền.

## 11. Percentile ties — nói quá

| Phát biểu | Đánh giá |
|---|---|
| RC curve bị **invalid**? | **Không.** AURC hoàn toàn xác định khi có ties nếu dùng quy ước kỳ vọng dưới phép phá ties ngẫu nhiên |
| RC curve là **step function**? | Đúng, và đó là bản chất, không phải khuyết tật |
| Ties tạo nhiều mẫu cùng điểm? | Đúng. Hệ quả thật: **không cắt được đúng coverage 80%** nếu một khối ties vắt ngang mốc. Xử lý bằng nội suy hoặc phá ties ngẫu nhiên — tầm thường |
| Average rank là cách xử lý ties? | **Sai lệch.** Average rank là cách đúng để **dựng ECDF**, nhưng nó **không phá ties** — các mẫu bằng nhau vẫn nhận cùng giá trị |

**Phải làm cả hai:** average rank cho Φ, và quy ước kỳ vọng-ngẫu-nhiên cho AURC.

**Hệ quả thật sự thú vị:** ties **làm xấu AURC của phương pháp có ít giá trị phân biệt**, vì buộc phải chấp nhận cả khối mẫu trong đó có ca sai. Đó là lập luận ủng hộ Φ_M (nó phá ties của Φ_S) — nhưng là **kết quả cần đo**, không phải lý do sửa protocol.

> **HẠ CẤP xuống implementation detail.** Chẩn đoán duy nhất cần: đếm số giá trị điểm phân biệt (unique score) của mỗi phương pháp, ghi phụ lục. Chữ "gãy" trong phản hồi cũ là nói quá.

## 12. Spearman ρ > 0,8 ⇒ suy biến: KHÔNG CÓ CƠ SỞ

> **TỰ BÁC BỎ HOÀN TOÀN.** Ngưỡng này được bịa ra, không có lý thuyết nào chống lưng.

**Vì sao sai:** selective classification **không quan tâm đến thứ hạng toàn cục**. Nó chỉ quan tâm liệu các ca SAI có bị đẩy xuống đáy bảng xếp hạng hay không.

Hai điểm số có thể tương quan hạng 0,95 trên toàn bộ dữ liệu mà vẫn bất đồng đúng ở 40 ca sai tự tin cao — và 40 ca đó quyết định toàn bộ AURC. Spearman toàn cục bị chi phối bởi khối đa số ca dễ, tức bởi phần dữ liệu **không liên quan** đến câu hỏi.

Tương tự, α\* → 0 hoặc 1 **không suy ra từ ρ cao**; nó suy ra từ việc một tín hiệu áp đảo *trong vùng lỗi*.

### 12.1 Chẩn đoán tối thiểu nhưng đủ mạnh (thay cho Spearman)

1. **ΔAURC có CI bootstrap ghép cặp** giữa α=1, α=0, α=α\*.
   → Đây chính là phép kiểm định giá trị gia tăng. Nếu CI của ΔAURC(α\* vs α=1) không chứa 0, Φ_M có giá trị. Không cần gì thêm cho tuyên bố chính.

2. **AUROC có điều kiện ở vùng tự tin cao**: trong nhóm mẫu có S thuộc decile cao nhất, Φ_M tách đúng/sai tốt đến đâu?
   → Đo trực tiếp cơ chế bài báo tuyên bố ("bắt lỗi tự tin thái quá"). **Đây là con số quan trọng nhất trong cả bài.**

3. **Hình scatter (Φ_S, Φ_M) tô màu đúng/sai.**
   → Rẻ, thuyết phục reviewer hơn mọi hệ số tương quan.

Spearman vẫn báo cáo được, nhưng chỉ như số liệu mô tả trong phụ lục.

---

# PHẦN IV — AUDIT ĐỀ XUẤT D

## 13. Đổi tên EMD-Gate

**Có phải methodological inconsistency không?** Không — thuần túy vấn đề đặt tên, không ảnh hưởng tính đúng đắn. Nhưng acronym mà chữ cái không khớp thành phần là thứ reviewer thật sự bắt lỗi, và tạo ấn tượng bài viết chắp vá.

**Điều KHÔNG được làm:** nhét lại thành phần Energy vào R(x) chỉ để cứu cái tên. Đó là để cái đuôi vẫy con chó.

**Cách xử lý có nguyên tắc:** Energy đã nằm sẵn trong danh sách baseline. Để ablation quyết định:
- Nếu thêm Φ_E vào tổ hợp lồi ba thành phần cải thiện AURC trên val → giữ, tên vẫn đúng.
- Nếu không → bỏ và đổi tên.

> **Tên đi sau thực nghiệm, không đi trước.**

**Nguyên tắc đặt tên (chưa đặt tên cụ thể):** acronym phải khai triển thành **đúng tập thành phần có mặt trong công thức cuối cùng**, và không chứa từ mô tả một cơ chế mà bài không cài đặt. Quyết định sau Bước 5.

## 14. Horizontal flip — quá quả quyết

| Câu hỏi | Phân tích | Hướng |
|---|---|---|
| Patch răng có còn thông tin trái/phải? | Có, một phần. Hình thái thân răng bất đối xứng (gờ bên gần/xa, điểm tiếp xúc lệch). Nhưng răng hàm trên phải lật ngang trông giống răng hàm trên trái — **một mẫu có thật trong dữ liệu** | Ủng hộ flip |
| Biểu hiện bệnh lý phụ thuộc trái/phải? | Không ở mức nhãn. 4 lớp không phân biệt bên; sâu mặt gần hay xa đều là Caries. Flip bảo toàn nhãn | Ủng hộ flip |
| Tọa độ/hướng còn trong crop? | Hướng dọc còn; hướng ngang mất ngữ cảnh cung hàm | Trung tính |
| Flip có đổi biểu hiện bệnh lý? | **Phản biện thật sự, đã bị bỏ qua:** chụp panorama có hình học bất đối xứng — ghost image xuất hiện phía đối diện và cao hơn, độ phóng đại đổi theo vị trí trong lớp cắt, vệt nhiễu kim loại có hướng lan đặc trưng. Train bất biến với lật ngang có thể **xóa đúng tín hiệu hướng mà stress-test nhiễu kim loại dựa vào** | Chống flip |

> **QUYẾT ĐỊNH: PILOT ABLATION trên val** — không cấm cũng không cho mặc định.
> **Ràng buộc thêm:** không dùng TTA (test-time augmentation) cho bất kỳ phương pháp nào — TTA thay đổi phân phối điểm tin cậy và phá vỡ tính công bằng giữa baseline.
> Rút lại phát biểu "bỏ quy tắc cấm lật ngang".

## 15. Relative Mahalanobis — bắt buộc

RMD (Ren et al., 2021) = D²_k(z) − D²_background(z), trừ đi khoảng cách tới một Gaussian chung không phụ thuộc lớp. Nó được thiết kế **chính xác để sửa điểm yếu của Mahalanobis thô trên near-OOD**.

Thành phần Φ_M của plan là min_k D² — **đúng cái đại lượng RMD được sinh ra để cải tiến** — và bối cảnh là near-distribution. Khả năng RMD một mình đánh bại Φ_M một mình là **thực sự cao**.

> **Verdict: baseline BẮT BUỘC, không phải nice-to-have.** Bỏ nó thì reviewer có lý do chính đáng yêu cầu bổ sung ở vòng revision.

**Chi phí:** một Gaussian toàn cục khớp trên train features, ~10 dòng code, tái sử dụng hạ tầng Σ đã có. Vào đúng một cột trong bảng baseline, **không đụng research question**.

## 16. Tufts external validation — RÚT LẠI

**Phản biện:**

- Tufts Dental Database **không có nhãn chẩn đoán 4 lớp tương thích DENTEX** → không thể validate bộ phân loại.
- Gọi nó là *"external validation cho selective pathology classification"* là **overclaim**; reviewer đọc kỹ sẽ bắt ngay → gây hại nhiều hơn lợi.
- Cùng lắm nó là artifact/domain-shift stress test.

**Về scope:** chuỗi *"DENTEX disease classification + selective classification + near-distribution detection + clinical artifact benchmark + external dataset"* là **đúng định nghĩa scope creep**. Thêm Tufts kéo theo tiền xử lý mới, một vòng annotation mới, một mục kết quả mới — ước tính **2–3 tuần** — để đổi lấy một đoạn văn reviewer sẽ đánh dấu là claim yếu.

**Điều quan trọng nhất:** động cơ của đề xuất cũ là *"để nhắm CMPB Q1"* — tức thêm dataset cho bài trông sang hơn. Đó là lý do không chấp nhận được.

> 🚩 **SCOPE CREEP — BÁC BỎ. Rút lại đề xuất của chính mình.**

**Thay thế:** khai báo trung thực giới hạn single-center/single-dataset trong mục Limitations, nêu external validation là hướng tương lai (một câu). Nhiều bài CMPB được nhận với giới hạn này khi phần đánh giá nội bộ chặt chẽ.

## 17. Protocol chuyên gia với một bác sĩ

| Câu hỏi | Trả lời |
|---|---|
| Một expert có được gọi là inter-rater reliability? | **Không, tuyệt đối không.** Inter-rater đòi hỏi ≥2 người đọc độc lập |
| Test-retest đo gì? | **Intra-rater reliability.** Thống kê vẫn là Cohen's κ (hai lần đọc của cùng một người); **cái thay đổi là tuyên bố, không phải công thức**. Phải viết rõ "intra-rater κ" |
| Khoảng cách hai lần đọc? | Quy ước 2–4 tuần. Với patch nhỏ dễ nhớ hơn phim toàn cảnh → **tối thiểu 3 tuần** |
| Blinded re-annotation? | **Có, ba lớp mù:** mù với nhãn lần một; thứ tự trình bày xáo lại; và **mù với output của model** |
| Guideline khóa trước hay viết sau? | **Bắt buộc khóa trước** |

### 17.1 Vì sao phải mù với output của model

Nếu bác sĩ gán nhãn cạm bẫy trên những ca model đã đoán sai, tập nhãn trở thành **hàm của model** → toàn bộ lập luận thành vòng tròn. Plan hiện tại không nói điều này.

### 17.2 Quy trình guideline tối thiểu

```
1. Viết guideline v1
   (định nghĩa cervical burnout; ranh giới phân biệt với sâu cổ răng thật;
    tiêu chí vệt kim loại; vùng chồng lấn giải phẫu)
2. Đọc thử 15 patch để hiệu chỉnh
3. KHÓA guideline
4. Gán nhãn chính thức trên TOÀN BỘ tập test (không lọc trước)
5. Sau ≥3 tuần: đọc lại ngẫu nhiên 100 mục, mù ba lớp
6. Báo cáo intra-rater κ kèm CI 95%
```

Guideline viết sau khi nhìn dữ liệu là nhãn hậu nghiệm và không bảo vệ được.

### 17.3 Cải tiến rẻ chưa được nêu

Sự hiện diện của **vệt kim loại** là phán đoán thị giác, **không cần chuyên môn RHM** — một người đọc thứ hai được huấn luyện ngắn là đủ.

→ Có **inter-rater κ thật cho nhãn artifact**, trong khi bác sĩ độc quyền phụ trách phán đoán bệnh lý (đo bằng intra-rater). Vừa trung thực vừa mạnh hơn phương án một người.

---

# PHẦN V — NĂM VẤN ĐỀ MỚI BỊ BỎ SÓT

> **Ghi nhận trước:** những phần plan v2 đã **đúng** và không cần động tới — patient-level split; loại 3 hộp đa nhãn; một backbone chung cho các phương pháp hậu nghiệm; bỏ cam kết số liệu trước thực nghiệm; định vị novelty ở lâm sàng thay vì toán học; min_k thay cho D²_ŷ; Ledoit-Wolf; tổ hợp lồi đơn điệu. Đó là một nền vững.

## N1 — Coverage cấp răng không chuyển thành lợi ích cấp phim (NGHIÊM TRỌNG)

Gate từ chối **theo từng răng**. Nhưng quy trình lâm sàng ra quyết định **theo từng phim/bệnh nhân**: chỉ cần một răng bị từ chối, cả phim phải chuyển bác sĩ đọc.

```
~5 răng có nhãn / phim, coverage cấp răng = 80%
→ P(phim có ≥1 răng bị từ chối) = 1 − 0,8^5 ≈ 67%
```

Nghĩa là hệ thống "giữ lại 80%" trên giấy nhưng thực tế **chuyển 2/3 số phim** cho bác sĩ — lợi ích giảm tải gần như bằng không. Mọi tuyên bố ứng dụng lâm sàng sẽ sụp nếu reviewer y khoa làm phép tính này.

> **BẮT BUỘC:** báo cáo đường cong risk–coverage ở **cả hai cấp — răng và phim**. Coverage cấp phim = tỷ lệ phim mà mọi răng đều được chấp nhận.

Đây vừa là đóng góp trung thực, vừa là một hình vẽ chưa bài selective classification nha khoa nào có.

## N2 — AURC bị chi phối bởi phân bố lớp, không phải chất lượng gate

Plan đã nhận ra shortcut của lớp Impacted nhưng chỉ đề xuất *"báo cáo 3-class và 4-class"*. Vấn đề sâu hơn:

- Impacted (17%, accuracy ~99%) tạo khối lớn mẫu cực dễ ở đầu đường RC.
- Caries chiếm 62%.
- → **AURC phần lớn là hàm của tiên nghiệm lớp.** Hai phương pháp chỉ có thể khác nhau ở phần 4,5% Periapical cộng vùng biên Caries/Deep Caries.
- → Hai phương pháp rất khác nhau về cơ chế vẫn có thể cho AURC gần bằng nhau.

> **BỔ SUNG:** báo cáo thêm **class-balanced selective risk** (rủi ro trung bình theo lớp thay vì theo mẫu) song song AURC chuẩn.
> Giữ **4-class làm phân tích chính** để không đổi research question; 3-class và class-balanced là phân tích phụ **khai báo trước**, không phải chọn sau khi thấy kết quả.

## N3 — Ablation Center Loss mâu thuẫn với tuyên bố công bằng của chính plan

- Mục 4.2 nói mọi phương pháp dùng chung một backbone.
- Mục 4.3 lại train một backbone thứ hai có Center Loss.
- Plan **không nói bảng kết quả chính thuộc backbone nào.**

**Nguy cơ cụ thể:** nếu phương pháp đề xuất được báo cáo trên backbone có Center Loss (vốn nén cụm đặc trưng chặt lại, có lợi trực tiếp cho Mahalanobis) trong khi baseline chạy trên backbone CE thuần → toàn bộ bảng so sánh vô giá trị, và reviewer sẽ coi là cherry-picking.

> **QUY TẮC phải ghi vào plan:** bảng chính chạy trên **backbone CE thuần cho TẤT CẢ 8 phương pháp**. Center Loss chỉ xuất hiện trong một ablation riêng, trong đó **cả 8 phương pháp được chạy lại**.

## N4 — CLAHE là confound trực tiếp với đối tượng nghiên cứu

Plan liệt kê CLAHE như augmentation mặc định. Nhưng **cervical burnout chính là một vùng thấu quang tương phản thấp** — CLAHE tác động trực tiếp lên chính đặc trưng ảnh định nghĩa cạm bẫy mà bài báo lấy làm trung tâm. Nó có thể làm model bất biến với burnout, hoặc khuếch đại nó; cả hai đều làm kết quả stress-test không diễn giải được.

Thêm nữa, plan không phân biệt CLAHE là:
- **tiền xử lý** (phải áp dụng y hệt lúc test), hay
- **augmentation ngẫu nhiên** (làm thay đổi manifold huấn luyện).

Hai lựa chọn này cho hai hệ thống khác nhau.

> **BẮT BUỘC:** CLAHE là một **ablation tường minh (bật/tắt)**, không phải mặc định, và phải khai báo rõ vị trí trong pipeline.

## N5 — Resize 224×224 xóa độ phóng đại, tạo confound tương quan với lớp

- Patch răng hàm ≈ 200×220 px; patch răng cửa ≈ 80×150 px.
- Cả hai resize về 224×224 → **tỷ lệ phóng đại khác nhau 2–3 lần**, thông tin kích thước thật bị xóa.
- Kích thước patch tương quan mạnh với loại răng; loại răng tương quan mạnh với lớp (Impacted gần như luôn là răng hàm lớn).

**Rủi ro:** feature manifold một phần đang mã hóa **kích thước răng** chứ không phải bệnh lý → Mahalanobis sẽ "phát hiện" răng có kích thước bất thường.

**Kiểm tra rẻ và dứt khoát:** lưu kích thước patch gốc làm biến đồng hành, hồi quy điểm g(x) theo kích thước patch. Nếu R² đáng kể → phải xử lý (đưa kích thước vào như đặc trưng hiển, hoặc chuẩn hóa scale theo chiều rộng cung hàm thay vì theo từng hộp).

## N5b — Reproducibility

Plan chưa có chính sách seed, chưa lưu file phân chia fold.

> **Lưu `folds.json` cố định và seed cho mọi lần train.** Không có nó thì không ai — kể cả tác giả sau ba tháng — tái lập được kết quả.

---

# PHẦN VI — BẢNG TỔNG KẾT AUDIT

| Vấn đề | Phản hồi cũ nói | Kết luận sau audit | Quyết định |
|---|---|---|---|
| **5-fold CV** | Bắt buộc, vì so sánh 8 phương pháp thiếu power | Lý do sai (bỏ qua thiết kế ghép cặp). Nhưng vẫn cần cho lớp hiếm và nhóm cạm bẫy | **MODIFY** — giữ, đổi lý do; bắt buộc có val fold xoay vòng |
| **Bootstrap CI** | CI 95%, 1.000 lần resample | Đúng, nhưng thiếu điều kiện sống còn: phải **cluster ở cấp ảnh** và **ghép cặp** cho ΔAURC | **KEEP + SỬA** |
| **Deep Ensemble từ 5 fold** | "Gần như miễn phí" | **Sai, gây leakage** — model fold i đã train trên fold j | **REJECT** — chạy K=5 seed trên 1 fold, ghi rõ 5× chi phí |
| **Ngưỡng τ** | Không đề cập | Leakage tinh vi: Risk@coverage mặc định dùng oracle threshold trên test | **THÊM MỚI** — báo cáo cả τ hiệu chỉnh trên val |
| **Unlabeled pool → abstain rate** | Metric có nghĩa nếu ghim coverage | Không phải metric, chỉ mô tả. Đã lén coi unlabeled = negative | **REJECT** ở dạng cũ → thí nghiệm phụ, chỉ khi có nhãn chuyên gia |
| **Same-image unlabeled teeth** | Ưu tiên phương án này | Có chi phí ẩn: cần train tooth detector | 🚩 **SCOPE CREEP — REJECT** |
| **Kiểm tra overlap 2 subset** | Bắt buộc, "đòn chí mạng" nếu rời nhau | Bắt buộc vì rẻ, nhưng "chí mạng" là nói quá. Kiểm bằng discriminator AUC | **KEEP, hạ giọng** |
| **d = 64** | Bắt buộc vì N_k = 126 < 128 | **Điều kiện N_k > d không áp dụng** cho tied covariance + shrinkage | **REJECT** → d là siêu tham số chọn trên val; giảm chiều bằng **PCA hậu nghiệm** |
| **Percentile ties** | "Sẽ làm gãy RC curve" | Nói quá. RC vẫn hợp lệ; chỉ là chi tiết cài đặt | **HẠ CẤP** → implementation detail + đếm unique score |
| **Spearman ρ > 0,8 ⇒ suy biến** | Nêu như quy tắc | **Không có cơ sở**, tự đặt ra | **REJECT** → thay bằng ΔAURC ghép cặp + AUROC có điều kiện ở decile tự tin cao |
| **Đổi tên EMD-Gate** | Nên đổi vì không còn Energy | Không phải lỗi phương pháp, nhưng reviewer sẽ bắt | **HOÃN** — để ablation quyết định |
| **Horizontal flip** | Nên bỏ lệnh cấm | Quá quả quyết — hình học panorama bất đối xứng, có thể xóa tín hiệu hướng của vệt kim loại | **MODIFY** → pilot ablation trên val; cấm TTA |
| **Relative Mahalanobis** | Nên thêm | Đúng, và mạnh hơn: là đối thủ trực tiếp của Φ_M trên near-distribution | **KEEP — nâng lên bắt buộc** |
| **Tufts external** | Nên cân nhắc | Overclaim + động cơ sai (nhắm tier tạp chí) | 🚩 **SCOPE CREEP — REJECT**, rút lại |
| **Expert protocol** | 1 expert → test-retest | Đúng hướng, thiếu 3 điều kiện: mù với output model, guideline khóa trước, khoảng cách ≥3 tuần. Thêm: người đọc thứ 2 cho nhãn artifact | **KEEP + BỔ SUNG** |

---

# PHẦN VII — FINAL RECOMMENDED PROTOCOL

> Chỉ giữ thay đổi thực sự cần. **Research question và tiêu đề không đổi**, trừ một cụm từ làm rõ phạm vi.

## 7.1 Dữ liệu

- 705 ảnh, **3.523 hộp** sau khi loại 3 hộp đa nhãn.
- **5-fold ở cấp ảnh**, mỗi fold: 3 train / 1 val / 1 test xoay vòng.
- Lưu `folds.json` và seed cố định.

## 7.2 Backbone

- ResNet-50, **CE thuần**, một model/fold, AMP.
- ~12 phút/fold trên T4 → **~1 giờ** cho toàn bộ phần hậu nghiệm.

## 7.3 Đặc trưng & manifold

- Feature 2048-d → **PCA khớp trên train folds**.
- d ∈ {32, 64, 128, 256} chọn theo AURC(val).
- μ_k và Σ (tied, Ledoit-Wolf) trên **train folds**.
- **Không đụng backbone.**

## 7.4 Gate

- Φ_S, Φ_M bằng ECDF **average-rank** trên val fold.
- α quét lưới 0–1 bước 0,05 trên val.
- τ hiệu chỉnh trên val theo coverage mục tiêu.

## 7.5 Baselines (8 cột, chung một backbone)

| # | Phương pháp | Ghi chú |
|---|---|---|
| 1 | MSP | Hendrycks & Gimpel, 2017 |
| 2 | Temperature Scaling | Guo et al., 2017 |
| 3 | MC-Dropout (T=20) | Gal & Ghahramani, 2016 |
| 4 | Mahalanobis (min_k) | Lee et al., 2018 |
| 5 | **Relative Mahalanobis** | Ren et al., 2021 — **bắt buộc** |
| 6 | Energy | Liu et al., 2020 |
| 7 | ViM | Wang et al., 2022 |
| 8 | **Phương pháp đề xuất** | R_α convex gating |
| (9) | Deep Ensembles | **Fold 1, K=5 seed**, đánh dấu rõ 5× chi phí (~50 phút thêm) |

## 7.6 Metrics

- AURC, E-AURC, Err-AUROC
- Risk@coverage {70, 80, 90} — **hai phiên bản**: oracle và τ-hiệu-chỉnh-trên-val (kèm coverage thực đạt)
- **Coverage cấp phim song song cấp răng** (N1)
- **Class-balanced selective risk** (N2)

## 7.7 Thống kê

- **Cluster bootstrap cấp ảnh**, 1.000 lần, **ghép cặp cho ΔAURC**
- So sánh chính **khai báo trước**: đề xuất vs MSP; đề xuất vs ViM
- Các so sánh còn lại là mô tả; hiệu chỉnh Holm nếu tuyên bố ý nghĩa thống kê trên toàn bảng

## 7.8 Ablation (chạy trên fold 1, ~3 lần train thêm)

- α = 1 / α = 0 / α = α\*
- **CLAHE bật–tắt** (N4)
- **Flip bật–tắt**
- **Center Loss bật–tắt, với cả 8 phương pháp chạy lại** (N3)
- Lưới d

## 7.9 Stress-test lâm sàng (thí nghiệm CHÍNH, không cần hạ tầng mới)

- Nhóm con răng **có nhãn bệnh** kèm cạm bẫy: vệt kim loại, cervical burnout, chồng lấn giải phẫu.
- Gán nhãn trên **toàn bộ tập test**, **mù với output model**.
- Guideline **khóa trước**, sau 15 patch hiệu chỉnh.
- **Intra-rater κ** trên 100 mục đọc lại sau ≥3 tuần.
- **Người đọc thứ hai** (không cần chuyên khoa) cho riêng nhãn artifact → **inter-rater κ thật**.

## 7.10 Pool không nhãn (thí nghiệm PHỤ, tùy chọn)

- Chỉ làm nếu bác sĩ đọc được ~300 răng mẫu.
- Kèm kiểm tra dịch chuyển subset bằng **discriminator AUC**.
- Gọi đúng tên: **unlabeled / near-distribution stress pool**. **Không bao giờ** gọi là OOD hay healthy.

## 7.11 Kiểm tra confound bắt buộc trước khi train

- Hồi quy g(x) theo **kích thước patch gốc** (N5).

## 7.12 Tổng compute

```
5  (fold)      + 4  (ensemble fold 1) + 3  (ablation)
= 12 lần train ≈ 2,5–3 giờ trên T4
→ chia được thành nhiều phiên Colab
```

---

## 7.13 Chỉnh sửa duy nhất về phạm vi

Thêm **"in Differential Diagnosis"** hoặc **"among Pathological Teeth"** vào tiêu đề, để cụm *"Under Anatomical and Metallic Artifacts"* không bị đọc thành lời hứa về false positive trên răng lành — thứ mà thiết kế này không đo và, sau audit này, **không nên cố đo**.

**Đây không phải đổi research question.** Nó là làm rõ phạm vi vốn đã có.

---

## BƯỚC TIẾP THEO

`00_sanity_checks.py` — bốn kiểm tra chặn đường, chạy **trước khi cắt patch**:

1. Phân bố lớp theo từng fold
2. Discriminator AUC giữa hai subset (705 vs 634)
3. Hồi quy g(x) ~ kích thước patch
4. Đếm unique score
