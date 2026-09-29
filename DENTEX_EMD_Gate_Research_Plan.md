# MASTER RESEARCH & ENGINEERING PLAN: SELECTIVE DENTAL AI (EMD-GATE) TRÊN DENTEX
*(Phiên bản cập nhật toàn diện: Chuẩn hóa Toán học, Dữ liệu thực tế và Đảm bảo Tính khả thi Xuất bản)*

**Tên đề tài chính thức (Đã hiệu chỉnh):**
> **Selective Classification for Dental Panoramic Radiographs: Mitigating Overconfident Diagnostic Failures Under Anatomical and Metallic Artifacts**  
> *(Phân loại Chọn lọc trên Phim X-quang Toàn cảnh: Giảm thiểu Sai sót Tự tin Thái quá do Cạm bẫy Giải phẫu và Nhiễu Kim loại)*

**Mục tiêu xuất bản thực tế:** Computer Methods and Programs in Biomedicine (CMPB - Q1, IF ~7.0) / Diagnostics (MDPI - Q2) / MICCAI UNSURE Workshop (Uncertainty for Safe Patient Registration and Analysis).

---

## 1. PHÂN TÍCH BẢN CHẤT DỮ LIỆU DENTEX & KIỂM CHỨNG TẠI BƯỚC 0

Từ khảo sát thực tế và kiểm tra trực tiếp mã nguồn file nhãn COCO gốc (`train_quadrant_enumeration_disease.json` và `validation_triple.json`):

### 1.1. Phân bố Nhãn Bệnh lý Chính thức (Official Categories & Data Audit)
Hệ thống nhãn chẩn đoán bệnh lý chính thức trong DENTEX gồm 4 lớp (ID quy chuẩn trong COCO annotation):
1.  **`0_Impacted`** (Răng khôn / răng ngầm kẹt trong xương): **604 mẫu** (17.1%)
2.  **`1_Caries`** (Sâu men / ngà nông): **2,189 mẫu** (62.0%)
3.  **`2_Periapical_Lesion`** (Tổn thương / nang quanh chóp): **158 mẫu** (4.5%) — *Mất cân bằng cực đoan!*
4.  **`3_Deep_Caries`** (Sâu ngà sâu / sát tủy): **578 mẫu** (16.4%)

*   **Kiểm tra Multi-label:** Trong 3,526 vị trí hộp khác nhau, chỉ có 3 hộp mang 2 nhãn đồng thời (0.09%). Do đó, giả định Single-label là hợp lệ; ta loại bỏ 3 hộp dị biệt này khỏi tập huấn luyện để tránh làm nhiễu nhãn.
*   **Vấn đề Shortcut của lớp `Impacted`:** Răng ngầm thường có vị trí đặc thù (góc hàm, vùng tam giác hậu hàm, mọc lệch/nằm ngang trong xương), mô hình có thể đạt accuracy ~99% dễ dàng từ hình dạng và tọa độ crop. Cần báo cáo kết quả tổng thể và kết quả bóc tách riêng (3-class vs 4-class) để tránh làm loãng độ đo rủi ro.

### 1.2. Giải quyết Mâu thuẫn Tiêu đề & Thiết kế: Tích hợp Unlabeled Tooth Pool
*   **Vấn đề:** Các cạm bẫy kinh điển (thấu quang cổ răng - *cervical burnout*, vệt kim loại từ mối hàn amalgam/mão, lỗ cằm, xoang hàm) chủ yếu gây ra **False Positive trên RĂNG LÀNH**. Nếu chỉ nhận patch của răng đã có nhãn bệnh, toàn bộ failure mode quan trọng nhất này bị bỏ sót ngoài phạm vi nghiên cứu.
*   **Giải pháp (Phương án 1):** Tận dụng subset `quadrant_enumeration` của DENTEX:
    *   Tập `quadrant_enumeration` có **18,095 hộp răng** trên 634 ảnh panorama được đánh số thứ tự từ 1 đến 8 ở 4 cung hàm nhưng **không có nhãn chẩn đoán bệnh**.
    *   **Thiết kế:**
        *   Tập `Train` (4 classes): Chỉ huấn luyện trên 4 lớp bệnh lý có ground truth xác nhận.
        *   Tập `Stress-test / Calibration Pool`: Trích xuất một tập răng từ subset `quadrant_enumeration`, khai báo minh bạch là **Unlabeled Teeth Pool (In-distribution / Near-distribution Negative Teeth)** để đánh giá năng lực từ chối (Abstain) của Gate trước các ca răng lành mang cạm bẫy giải phẫu hoặc vệt kim loại.

### 1.3. Chuẩn hóa Lề Cắt Patch (Adaptive Margin: 5% - 8%)
*   **Kiểm chứng Overlap:** Khi pad 15%, tỷ lệ overlap giữa các răng có bệnh tăng vọt từ **15.45% lên 26.15%** (hơn 1/4 số răng bị chồng lấn, dễ kéo tổn thương của răng lân cận vào patch).
*   **Quy chuẩn mới:** Sử dụng lề mở rộng $5\%$ (tối đa $8\%$) để vừa giữ được trọn vẹn ranh giới men - ngà (cổ răng) và chóp cuống răng, vừa kiểm soát tỷ lệ overlap ở mức an toàn (~20%).

### 1.4. Nguyên tắc Phân chia Tập Dữ liệu (Strict Patient-level Split)
*   Tập dữ liệu huấn luyện DENTEX gồm 705 ảnh panorama. Phân chia theo tỷ lệ **70% Train (493 ảnh) / 15% Val (106 ảnh) / 15% Internal Test (106 ảnh)**.
*   Việc phân chia **bắt buộc thực hiện ở cấp độ Bệnh nhân / Ảnh Panorama (`image_id`)** trước khi cắt patch, tuyệt đối không xáo trộn ngẫu nhiên ở cấp độ patch để triệt tiêu nguy cơ rò rỉ dữ liệu (Patient-level data leakage).

---

## 2. QUY TRÌNH TIỀN XỬ LÝ (PATCH EXTRACTION PIPELINE)

```
              Ảnh Panorama Gốc (~2000x1000) + File Annotation COCO JSON
                                         │
                                         ▼
           1. Patient-level Split: Chia 705 ảnh thành Train / Val / Test
                                         │
                                         ▼
           2. Đọc Bounding Box gốc: [x_min, y_min, width, height]
                                         │
                                         ▼
           3. Mở rộng lề thích ứng an toàn (Adaptive Padding Margin = 5% - 8%):
              • x_new = max(0, x - 0.06 * w)
              • y_new = max(0, y - 0.06 * h)
              • w_new = min(W_img, w + 0.12 * w)
              • h_new = min(H_img, h + 0.12 * h)
                                         │
                                         ▼
           4. Cắt (Crop) mẩu ảnh răng và Resize chuẩn về 224x224 pixels
                                         │
                                         ▼
           5. Lưu trữ theo Thư mục Chuẩn:
              ├── train/ (4 Official Classes)
              ├── val/   (4 Official Classes)
              ├── test/  (4 Official Classes)
              └── stress_test/
                  ├── unlabeled_teeth/   (Cắt từ quadrant_enumeration)
                  ├── metallic_artifacts/ (Ca có vệt Amalgam / Mão sứ - kim loại)
                  └── cervical_burnout/   (Ca có thấu quang cổ răng sinh lý)
```

---

## 3. KIẾN TRÚC TOÁN HỌC & ĐỘNG CƠ SELECTIVE GATING (EMD-GATE)

```
                            KIẾN TRÚC TOÁN HỌC EMD-GATE
                            
                      Tooth Patch x (224x224)
                                 │
                                 ▼
                     Feature Backbone h_θ (ResNet-50)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Feature Vector z ∈ ℝᵈ              Logits f(x) ∈ ℝ⁴
                 │                               │
                 ▼                               ▼
     Dimensionality Reduction           Softmax Confidence
      (Projection xuống d=128)             S(x) = max Softmax
                 │                               │
                 ▼                               │
       Class-Conditional Means                   │
          μ_k & Shrunk Cov Σ                     │
                 │                               │
                 ▼                               │
        Negative Mahalanobis                     │
      g(x) = - min_k D_k²(z)                     │
                 │                               │
                 ▼                               ▼
        Percentile Mapping              Percentile Mapping
       (trên Validation Set)           (trên Validation Set)
            Φ_M(g(x))                       Φ_S(S(x))
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                      Convex Gating Mechanism
                 R_α(x) = α·Φ_S(S) + (1-α)·Φ_M(g)
                                 │
                                 ▼
                         Reliability Gate
                         ┌───────┴───────┐
                         ▼               ▼
                    R_α(x) ≥ τ      R_α(x) < τ
                         │               │
                      ACCEPT          ABSTAIN
                    (Chấp nhận)   (Chuyển Bác sĩ)
```

### 3.1. Các Khắc phục Toán học Cốt lõi

#### A. Loại bỏ Underflow số học của $M(x)$
*   **Vấn đề cũ:** Công thức $M(x) = \exp(-0.5 D^2)$ với $d=2048$ dẫn tới $D^2 \sim 2048$, $\exp(-1024)$ underflow về $0.0$ tuyệt đối trong chuẩn IEEE 754 float32 và float64.
*   **Giải pháp:** Bỏ hàm $\exp$. Sử dụng trực tiếp giá trị log-density / **Negative Mahalanobis Distance**:
    $$g(x) = - \min_{k \in \{0..3\}} (z - \mu_k)^\top \Sigma_{\text{shrunk}}^{-1} (z - \mu_k)$$

#### B. Khắc phục Suy biến Ma trận Hiệp phương sai ($p \gg n$) & Tâm tham chiếu Centroid
*   **Vấn đề cũ:** Lớp `Periapical Lesion` chỉ có $N=158$ mẫu trong khi $d=2048$, ma trận $\Sigma$ chắc chắn suy biến, không thể khả nghịch. Ngoài ra, đo khoảng cách tới tâm $\mu_{\hat{y}}$ của lớp dự đoán sẽ sai lệch hoàn toàn khi mô hình đưa ra dự đoán sai $\hat{y}$.
*   **Giải pháp:**
    1.  **Giảm chiều:** Đưa vector đặc trưng qua tầng chiếu tuyến tính hoặc Adaptive Average Pooling về chiều $d = 128$ (đảm bảo $N_k > d$).
    2.  **Ledoit-Wolf Shrinkage:** Ước lượng ma trận hiệp phương sai gộp (Tied Covariance Matrix) với hệ số co rút tối ưu $\beta^*$:
        $$\Sigma_{\text{shrunk}} = (1 - \beta^*) \Sigma_{\text{emp}} + \beta^* \left( \frac{\text{Tr}(\Sigma_{\text{emp}})}{d} \right) I$$
    3.  **Tâm tham chiếu $\min_k D_k^2$:** Đo khoảng cách tới tâm cụm *gần nhất* trong toàn bộ không gian đặc trưng để kiểm tra xem mẫu có thuộc vùng nâng đỡ (manifold support) của bất kỳ lớp bệnh nào hay không.

#### C. Đảm bảo Tính Đơn điệu Tuyệt đối của Hàm Tin cậy $R(x)$
*   **Vấn đề cũ:** Hàm cũ $R = S(1 - S + \Phi)$ có đạo hàm âm khi $S > 0.525$, dẫn đến việc mô hình càng tự tin thì điểm tin cậy lại càng tụt dốc. Phép trừ $S - \Phi$ là phép trừ ad-hoc giữa xác suất và phân vị.
*   **Giải pháp (Percentile Mapping & Convex Combination):**
    1.  Chuẩn hóa cả hai tín hiệu về cùng thang đo phân vị thực nghiệm $[0, 1]$ dựa trên tập Validation:
        *   $\Phi_S(S(x)) = \frac{1}{N_{\text{val}}} \sum_{j=1}^{N_{\text{val}}} \mathbb{I}(S(x_j) \le S(x))$: Phân vị độ tự tin Softmax.
        *   $\Phi_M(g(x)) = \frac{1}{N_{\text{val}}} \sum_{j=1}^{N_{\text{val}}} \mathbb{I}(g(x_j) \le g(x))$: Phân vị khoảng cách Manifold.
    2.  Hàm độ tin cậy mới kết hợp đơn điệu:
        $$R_\alpha(x) = \alpha \cdot \Phi_S(S(x)) + (1 - \alpha) \cdot \Phi_M(g(x))$$
        *   Đảm bảo tính đơn điệu nghiêm ngặt: $\frac{\partial R_\alpha}{\partial \Phi_S} = \alpha > 0$ và $\frac{\partial R_\alpha}{\partial \Phi_M} = 1 - \alpha > 0$.
        *   Tham số $\alpha \in [0, 1]$ được hiệu chỉnh trên tập Validation để tối ưu hóa diện tích dưới đường cong rủi ro (AURC).

---

## 4. QUY TRÌNH ĐÁNH GIÁ CHUẨN MỰC (SELECTIVE CLASSIFICATION)

Tuân thủ chặt chẽ lý thuyết Selective Classification của Chow (1970) và Geifman & El-Yaniv (2017, 2019):

### 4.1. Các Độ đo Cốt lõi
1.  **Đường cong Risk-Coverage (RC Curve):**
    *   Với mỗi ngưỡng quyết định $\tau \in [0, 1]$, tỷ lệ chấp nhận (Coverage) và rủi ro còn lại (Selective Risk / Error Rate trên tập được chấp nhận):
        $$\text{Coverage}(\tau) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(R_\alpha(x_i) \ge \tau)$$
        $$\text{Risk}(\tau) = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i \ne y_i \land R_\alpha(x_i) \ge \tau)}{\sum_{i=1}^N \mathbb{I}(R_\alpha(x_i) \ge \tau)}$$
2.  **AURC (Area Under the Risk-Coverage Curve) & E-AURC (Excess AURC):**
    *   AURC càng thấp, hệ thống càng từ chối hiệu quả các ca dự đoán sai.
    *   E-AURC đo lường khoảng cách thừa giữa phương pháp và bộ chọn lý tưởng (Oracle selector).
3.  **Selective Risk tại các mốc Coverage Cố định (Fixed Coverage Risk):**
    *   Báo cáo tỷ lệ sai sót thực tế khi hệ thống giữ lại: **Coverage = 70%, 80%, 90%**.
    *   Loại bỏ hoàn toàn việc cam kết số ảo trước thực nghiệm ("từ >20% xuống <3%").
4.  **Err-AUROC (Failure Detection AUROC):** Năng lực phân định nhị phân giữa mẫu đoán Đúng ($y = \hat{y}$) và mẫu đoán Sai ($y \ne \hat{y}$).

### 4.2. Danh mục Phương pháp So sánh (Baselines Matrix)
Tất cả các phương pháp hậu nghiệm đều được đánh giá trên **cùng một Backbone duy nhất** để đảm bảo tính công bằng (Fairness):
1.  **MSP (Maximum Softmax Probability):** Baseline chuẩn (Hendrycks & Gimpel, 2017).
2.  **Temperature Scaling (TS):** Hiệu chỉnh nhiệt độ Softmax trên tập Val (Guo et al., 2017).
3.  **Monte Carlo Dropout (MC-Dropout):** Đo độ bất định biểu kiến với $T=20$ stochastic passes (Gal & Ghahramani, 2016).
4.  **Deep Ensembles:** Huấn luyện $K=5$ mô hình độc lập (Lakshminarayanan et al., 2017).
5.  **Mahalanobis Baseline:** Phương pháp của Lee et al. (NeurIPS 2018).
6.  **Energy-based OOD:** Phương pháp của Liu et al. (NeurIPS 2020).
7.  **ViM (Virtual-logit Matching):** Wang et al. (CVPR 2022).
8.  **EMD-Gate (Đề xuất):** Gating kết hợp phân vị đơn điệu $R_\alpha(x)$.

### 4.3. Thiết kế Thực nghiệm Ablation Study
*   **Ablation 1 (Backbone Loss Impact):** So sánh hiệu năng của EMD-Gate khi backbone được huấn luyện bằng $\mathcal{L}_{\text{CE}}$ đơn thuần vs $\mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{Center}}$ (chứng minh Gate hoạt động tốt độc lập với loss).
*   **Ablation 2 (Component Decomposition):**
    *   Chỉ dùng Softmax ($\alpha = 1.0$).
    *   Chỉ dùng Manifold Distance ($\alpha = 0.0$).
    *   Kết hợp hai thành phần ($\alpha = \alpha^*$).

---

## 5. NOVELTY LÂM SÀNG & PROTOCOL ĐÁNH GIÁ CHUYÊN GIA

Đóng góp khoa học cốt lõi của nghiên cứu được định vị ở **tính ứng dụng lâm sàng và bộ benchmark kiểm thử cạm bẫy thực tế**:

1.  **Xây dựng Bộ Stress-test Benchmark Cạm bẫy X-quang RHM:**
    *   Trích xuất các ca răng lành và răng bệnh có sự hiện diện của: (1) Thấu quang cổ răng sinh lý (*cervical burnout*), (2) Vệt cản quang/nhiễu tán xạ kim loại (*metallic artifact* từ amalgam/mão sứ kim loại), (3) Vùng cản quang giải phẫu chồng lấn (lỗ cằm, xoang hàm trên).
2.  **Protocol Đánh giá Chuyên gia Độc lập (Inter-Rater Reliability):**
    *   2 Bác sĩ Răng Hàm Mặt độc lập đọc và phân loại các ca cạm bẫy giải phẫu / nhiễu kim loại.
    *   Tính toán chỉ số tương đồng liên chuyên gia **Cohen's Kappa ($\kappa$)** để chứng minh độ tin cậy của tập nhãn cạm bẫy.
3.  **Tuyên bố Đóng góp (Paper Contribution):**
    *   Bài báo không tuyên bố phát minh một cấu trúc toán học OOD hoàn toàn mới (tránh so sánh bất lợi với ViM / Relative Mahalanobis trên lý thuyết thuần túy).
    *   Bài báo đóng góp một **Quy trình Phân loại Chọn lọc có căn cứ lâm sàng vững chắc, kèm bộ benchmark cạm bẫy nha khoa được chuẩn hóa và giải thuật Gating đơn giản, đơn điệu, hiệu quả cao trong việc ngăn chặn sai sót tự tin thái quá của AI**.

---

## 6. LỘ TRÌNH 5 BƯỚC TRIỂN KHAI KỸ THUẬT (MÃ NGUỒN)

1.  **Bước 1 (`01_extract_patches.py`):**
    *   Đọc nhãn COCO, thực hiện Patient-level Split (70:15:15).
    *   Loại bỏ 3 bounding box trùng lặp nhãn.
    *   Cắt patch với lề mở rộng an toàn 6% và lưu về cấu trúc thư mục chuẩn.
    *   Trích xuất pool răng không nhãn từ `quadrant_enumeration`.
2.  **Bước 2 (`02_dataset_loader.py`):**
    *   Xây dựng PyTorch Dataset, áp dụng Data Augmentation y khoa chuẩn (xoay nhẹ, điều chỉnh độ tương phản CLAHE, không lật ảnh làm đảo chiều răng).
3.  **Bước 3 (`03_train_backbone.py`):**
    *   Huấn luyện Backbone ResNet-50 với Cross-Entropy Loss (và nhánh phụ CE + Center Loss để làm ablation).
    *   Sử dụng Mixed Precision (FP16) trên Google Colab T4.
4.  **Bước 4 (`04_compute_manifold.py`):**
    *   Chiếu feature vector về $d=128$.
    *   Tính tâm cụm $\mu_k$ cho 4 lớp và ma trận hiệp phương sai co rút $\Sigma_{\text{shrunk}}$ theo thuật toán Ledoit-Wolf.
    *   Lưu các tham số vào `checkpoints/manifold_stats.npz`.
5.  **Bước 5 (`05_emd_gate_eval.py`):**
    *   Tính toán $\Phi_S$ và $\Phi_M$ trên tập Validation để tìm tham số tối ưu $\alpha^*$.
    *   Chạy đánh giá toàn bộ 8 baselines trên tập Internal Test và Stress-test Pool.
    *   Xuất biểu đồ Risk-Coverage Curves, bảng so sánh AURC, E-AURC và Risk @ Fixed Coverage.
