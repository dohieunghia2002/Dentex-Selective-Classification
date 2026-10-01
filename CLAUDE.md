# Dental_Research — DENTEX Selective Classification

Đề tài: *Selective Classification for Dental Panoramic Radiographs in Differential
Diagnosis: Mitigating Overconfident Failures Under Anatomical and Metallic Artifacts*.
Mục tiêu xuất bản: CMPB / IEEE JBHI / IEEE ISBI / Diagnostics / MICCAI UNSURE.

## Nguồn sự thật

`DENTEX_Research_Plan.md` (v6, khóa 2026-09-29) là **pre-registration**. Mọi tham số,
metric, baseline, ablation đã khóa — không sửa sau khi nhìn kết quả. Khi có mâu thuẫn
giữa file này và plan, **plan thắng**.

**Trước khi viết code hoặc bản thảo, luôn dùng skill `dentex-protocol`.** Skill đó là
bản nén các điều khoản bắt buộc; plan v6 là bản đầy đủ — đọc plan khi cần tra sâu
(công thức RMD §6.3, pseudo-code bootstrap §8.1, changelog v5→v6 §15).

Nếu tôi đề xuất một thay đổi vi phạm điều khoản đã khóa, **nói thẳng ra là vi phạm
điều nào** thay vì lặng lẽ làm theo. Mọi sai lệch ghi vào DEVIATION LOG (§14) kèm
ngày và lý do **trước khi** áp dụng.

## Đường dẫn thật — tên thư mục dùng GẠCH NỐI

```
DENTEX/training_data/quadrant-enumeration-disease/      ← subset DUY NHẤT của bài
    train_quadrant_enumeration_disease.json             ← annotation COCO
    xrays/                                              ← 705 ảnh panorama
DENTEX/training_data/quadrant_enumeration/xrays/        ← 634 ảnh, chỉ dùng cho pilot §11.2(a)
DENTEX/training_data/quadrant/xrays/                    ← 693 ảnh, KHÔNG dùng
DENTEX/validation_data/, DENTEX/test_data.zip           ← KHÔNG dùng (nhãn không công khai)
```

Plan viết `quadrant_enumeration_disease` (gạch dưới) nhưng **thư mục trên đĩa dùng gạch
nối**. Luôn dùng tên thư mục thật ở trên.

Dữ liệu đã giải nén đủ: `quadrant-enumeration-disease/xrays` có 705 file `train_*.png`.
`01_extract_patches.py` vẫn phải **assert đủ 705 ảnh** và dừng nếu thiếu — không im lặng
chạy trên tập con.

## Hằng số dữ liệu — `constants.json`, KHÔNG chép lại ở đây

> **Mọi hằng số dữ liệu nằm ở `constants.json` và chỉ ở đó.** File này cố tình **không**
> ghi lại các con số. Cần số nào thì mở `constants.json`; viết số vào code là sai.
> Nếu bất kỳ file nào — kể cả skill hay plan — mâu thuẫn với `constants.json` thì
> **`constants.json` đúng**, vì nó được đếm trực tiếp từ COCO gốc chứ không chép tay.

```python
import json
C = json.load(open("constants.json"))
C["images"]["total"]                  # tổng số ảnh
C["images"]["empty_image_ids"]        # loại khỏi mọi metric cấp ảnh (§7.3)
C["patches"]["total"]                 # đơn vị phân tích
C["classes"]["patch_level"]           # phân bố lớp — KHÔNG dùng annotation_level
C["split"]["params"]["random_state"]  # seed chia fold
```

Tái sinh sau bất kỳ thay đổi nào ở annotation nguồn:

```
python 00_sanity_checks.py --emit-constants
```

Lệnh này đếm lại từ COCO và ghi đè các con số, giữ nguyên phần diễn giải bằng chữ.
`00_sanity_checks.py` và `01_extract_patches.py` **dừng hẳn** nếu thiếu `constants.json`
— không quay về số viết cứng.

Bẫy hay gặp: bộ `604 / 2.189 / 158 / 578` là **cấp annotation**, không phải cấp patch.
Đơn vị phân tích của bài là **patch**. Dùng `C["classes"]["patch_level"]`.

**Hai sự thật đã kiểm chứng, tránh kết luận sai:**

- **Metadata ảnh chỉ có `height`, `width`, `id`, `file_name` — KHÔNG có patient ID.**
  Không thể kiểm tra rò rỉ ở cấp bệnh nhân; đơn vị chia fold là `image_id` (§3.1).
  Ghi hạn chế này vào Limitations, không suy diễn rằng mỗi ảnh là một bệnh nhân.
- **Tên file TRÙNG giữa các subset nhưng là ảnh KHÁC NHAU.** Cả hai subset đều đánh số
  lại từ `train_0.png`, nên 634/634 tên của `quadrant_enumeration` trùng với tên trong
  subset disease. **Mọi phép so trùng theo tên file hoặc image ID giữa hai subset đều
  vô nghĩa** — muốn so thật phải hash nội dung ảnh. §11.2(a) không đòi hỏi chứng minh
  hai subset disjoint; nó chỉ cần 15 patch lấy từ ngoài 705 ảnh.

Nhãn bệnh lý nằm ở `category_id_3` (0=Impacted, 1=Caries, 2=Periapical Lesion,
3=Deep Caries). `category_id_1` là quadrant, `category_id_2` là số thứ tự răng —
**không dùng cho bài này**.

## Ràng buộc vận hành

- **GPU: Colab free (Tesla T4).** Máy này không train. Claude Code viết script để tôi
  tự chạy trên Colab — **không tự chạy training, không tự chạy thí nghiệm nặng**.
- **Ngân sách: 11 lần train** (5+4+2, Center Loss bỏ theo §9.4): 5 chính + 4 Deep
  Ensembles + 2 ablation (CLAHE, flip). Slot 12 của `03_train_backbone.py` giữ chỗ, không
  dùng. Mọi đề xuất làm tăng con số này phải nêu rõ chi phí và hỏi trước.
- Checkpoint sau mỗi vòng CV (Colab free giới hạn phiên, GPU không cố định).
- Seed cố định cho: chia fold, khởi tạo model, shuffle dataloader, bootstrap.

## Năm điều khoản dễ vi phạm nhất khi viết code

1. **Không tồn tại hàm nào tính AURC bằng cách sort điểm THÔ của cả 5 fold chung một
   bảng.** Chỉ hai đường: per-fold (§7.1) và pooled-normalized (§7.6).
2. **`d = 64` là hằng số**, không nhánh nào chọn `d` từ dữ liệu (ngoài ablation §9.5).
3. **`03_train_backbone.py` không có vòng lặp nào quét LR, CLAHE hay flip** — đúng 5 lần train.
4. **Bootstrap resample ở cấp ẢNH**, giữ cấu trúc fold, ghép cặp giữa các phương pháp.
5. **AURC tích phân trên `[0,1]`** với mở rộng hằng số `r(c)=r_1` trên `(0, c_1]`.

`folds.json` sinh một lần rồi **commit, không sinh lại**.

## Các file khác trong thư mục

| File | Là gì |
|---|---|
| `DENTEX_Research_Plan.md` | **v6 — bản khóa, nguồn sự thật** |
| `DENTEX_EMD_Gate_Research_Plan.md` | Bản cũ. **Lỗi thời — không dùng làm căn cứ** |
| `constants.json` | **Nguồn sự thật DUY NHẤT cho hằng số dữ liệu** — dẫn xuất từ COCO |
| `DENTEX_Methodology_Audit.md` | Audit phương pháp luận dẫn tới v6 |
| `DENTEX_Naming_Audit.md` | Audit tên phương pháp — naming đang HOÃN theo §5.6 |
| `README.md` | README dự án (GitHub) |
| `DATASET_README.md` | README của **dataset DENTEX** |
| `PROGRESS.md` | Theo dõi tiến độ — cập nhật sau mỗi bước |
| `dentex-protocol.skill` | Bản đóng gói skill, không phải file nguồn |
| `*.pdf` | Paper tham khảo |
| `figures/` | Hình minh họa dataset |

## Cách làm việc

- Trả lời tôi bằng **tiếng Việt**; code, comment và docstring bằng **tiếng Anh**.
- Một khuyến nghị rõ ràng, không liệt kê nhiều phương án để tôi tự chọn.
- Ngắn gọn. Tôi đang tiết kiệm token.
