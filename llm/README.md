# Nghiên cứu LLM (công thức INLG 2025): setup và cách chạy

Package này train Gemma-7B + LoRA theo công thức INLG 2025 (ASQP dạng sinh code) trên **đúng các bộ dữ liệu của bảng kết quả chính**, rồi chấm điểm trên test. Chỉ model thay đổi, dữ liệu giữ nguyên.

Có 120 lần train: 8 nhánh × 3 bộ dữ liệu (Vi-AbSQA, ACOS Rest, ACOS Laptop) × 5 seed (42, 123, 2024, 7, 99).

| Nhánh | Dữ liệu train | Thứ tự quad |
|---|---|---|
| `none` | gold train (baseline) | gold |
| `prop` | gold + vote của teacher MvP + ADA joint (phương pháp đề xuất) | text |
| `copy` | gold + gold nhân bản cho bằng kích thước `prop` (đối chứng) | text |
| `eda`, `concat`, `ada`, `genda` | gold + file augment của phương pháp đó | gold |
| `rc` | file train đã ghép review của Review Combination, thay cho gold | gold |

Danh sách file của từng nhánh nằm ở `arms()` trong `llm/inlg.py`.

## 1. Yêu cầu

- **GPU:** một A100 80GB. Bộ nhớ đỉnh đo được khoảng 64–72 GB.
- **Python 3.10, 3.11 hoặc 3.12**, với **torch có CUDA cài sẵn** (đã test torch 2.8.0+cu128). Package không cài torch.
- **Ổ đĩa:** khoảng 20 GB cho weights của Gemma-7B, cộng vài GB cho output.
- **Không cần Internet trên server:** thư viện (transformers, peft, accelerate…) có sẵn dạng wheel trong package.
- **Thời gian:** khoảng 175 giờ A100 (7–8 ngày). Riêng `prop` và `copy` mất khoảng 100 giờ, vì mỗi run có 15–30 nghìn mẫu.

## 2. Chuẩn bị trên máy có Internet

**Package:** file `dist/asqp_llm_package.zip` đã build sẵn. Muốn build lại (ví dụ sau khi sửa code), chạy từ thư mục gốc repo:

```bash
bash llm/build_package.sh        # -> dist/asqp_llm_package.zip (code + dữ liệu + wheel Linux, khoảng 100 MB)
```

**Weights của model.** Gemma-7B là model gated, nên tài khoản Hugging Face phải chấp nhận license ở trang `google/gemma-7b` trước. Sau đó tải về một thư mục cache:

```bash
huggingface-cli login                       # dán token HF của bạn
huggingface-cli download google/gemma-7b --cache-dir /duong/dan/toi/models
```

Rồi chép `asqp_llm_package.zip` và thư mục `models` sang server.

Nếu weights đang nằm ở một thư mục thường (không phải cache HF), thì ở bước 3 dùng `export ASQP_MODEL=/duong/dan/toi/gemma-7b` thay cho `ASQP_CACHE_DIR`.

## 3. Setup trên server (một lần)

```bash
unzip asqp_llm_package.zip -d asqp_llm
cd asqp_llm
conda activate LLM                          # env có sẵn torch + CUDA
export ASQP_CACHE_DIR=/duong/dan/toi/models
bash llm/setup.sh
```

`setup.sh` tạo một venv riêng tên `.venv-asqp` và cài các thư viện đã ghim phiên bản vào đó, nên env `LLM` không bị đụng tới. Cuối cùng nó chạy **preflight**, in một loạt dòng kiểm tra:

- phiên bản thư viện;
- GPU;
- weights của model;
- số dòng và sha256 của từng file dữ liệu của mọi nhánh;
- chuyển đổi target qua lại cho cả hai thứ tự quad.

Tất cả đều phải là `PASS`. Muốn chạy lại preflight: `.venv-asqp/bin/python llm/inlg.py preflight`.

## 4. Chạy

Nên chạy thử toàn bộ chuỗi trên vài mẫu trước, khoảng 30–60 phút. Output nằm trong `outputs/llm_smoke`, không lẫn với run thật:

```bash
source .venv-asqp/bin/activate
python llm/pipeline.py run --smoke
```

Rồi chạy thật:

```bash
nohup python llm/pipeline.py run > run_all.log 2>&1 &
```

- **Thứ tự:** `none`, `prop`, `copy` trên cả 3 bộ trước, rồi đến 5 phương pháp literature. Nếu phải dừng sớm, phần so sánh chính vẫn đã xong.
- **Tiến độ:** `tail -f run_all.log`, hoặc `python llm/pipeline.py plan` để xem bước nào `done`, bước nào `todo`. Log của từng bước nằm ở `outputs/llm/_logs/`, thời gian chạy ở `outputs/llm/_status.tsv`.
- **Bị ngắt giữa chừng:** chạy lại đúng lệnh trên. Bước đã xong tự bỏ qua, run đang dở tiếp tục từ epoch cuối đã xong.
- **Chạy một phần:** `--only` nhận tiền tố tên bước, ngăn cách bằng dấu phẩy. Ví dụ `python llm/pipeline.py run --only vi_prop,vi_copy`, hoặc `--datasets rest --seeds 42,123`.

## 5. Gửi kết quả về

Khi chạy xong, pipeline tự tạo `results_llm_<máy>_<thời gian>.zip` ở thư mục gốc. File này có prediction, metric và config của mọi run, không có weights hay adapter. Gửi lại file đó là được. Muốn đóng gói lại bằng tay: `python llm/pipeline.py pack`.

## 6. Qwen2.5-14B (tùy chọn)

Công thức Qwen của notebook `LLM_QWen.ipynb` dùng prompt `json_format` và batch hiệu dụng 8:

```bash
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --cache-dir /duong/dan/toi/models   # trên máy có Internet
export ASQP_PROFILE=qwen
python llm/pipeline.py run          # output ở outputs/llm_qwen, không ghi đè kết quả Gemma
```

Chạy Qwen chậm hơn Gemma khoảng 1.3 lần.

## 7. Khi gặp lỗi

- **Preflight `FAIL` ở dòng phiên bản thư viện:** chưa kích hoạt `.venv-asqp`, hoặc `setup.sh` chưa chạy xong.
- **Preflight `FAIL` ở dòng `model weights`:** kiểm tra lại `ASQP_CACHE_DIR` (hoặc `ASQP_MODEL`), và kiểm tra weights đã tải đủ file `.safetensors` chưa.
- **Hết bộ nhớ GPU:** code tự chia nhỏ batch khi gặp OOM. Nếu vẫn lỗi, thêm `--micro_tokens 2048` (hoặc `1024`) vào lệnh `pipeline.py run`; tham số này được truyền xuống mọi bước train mà không đổi batch hiệu dụng.
- **Một bước bị `FAILED`:** xem 15 dòng cuối của log được in ra, hoặc mở log đầy đủ trong `outputs/llm/_logs/`. Các bước khác vẫn tiếp tục chạy.

Notebook gốc của nhóm INLG và những chỗ khác biệt so với chúng: xem `third_party/inlg2025/README.md` và docstring đầu `llm/inlg.py`.
