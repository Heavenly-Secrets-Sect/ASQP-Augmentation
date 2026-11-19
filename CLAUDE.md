# DataAugmentation (ASQP-Augmentation)

Nghiên cứu các phương pháp tăng cường dữ liệu (data augmentation) cho ASQP tiếng Việt (nhà hàng, laptop...), mô hình Paraphrase trên ViT5-base. Thư mục cục bộ tên `DataAugmentation`, remote GitHub là `Heavenly-Secrets-Sect/ASQP-Augmentation`. Repo **public** (session Kaggle clone đúng SHA không cần token), nên đừng đưa gì riêng tư vào đây. **Đọc `RUNNING.md` trước** (bố cục, giao thức, quy trình round và session); ghi chú nghiên cứu ở `notes/` (`README.md`, `DEVIATIONS.md`, `PAPER_NOTES.md`).

## Lệnh
- Thử nhanh trên CPU (khoảng 1 phút): `HF_HUB_OFFLINE=1 python3 src/train_paraphrase.py --output_dir /tmp/smoke --seeds 42 --limit 8 --epochs 2 --cpu`. Repo **không có bộ test tự động**; hãy chạy lệnh này (và `kaggle_queue.py list`) sau khi sửa `src/` hoặc `scripts/`.
- Xem tình trạng các round: `python3 scripts/kaggle_queue.py list`
- Lập kế hoạch một round: `python3 scripts/kaggle_queue.py plan --round <id>` (ghi `experiments/run_plan.json` và `sessions`, `est_hours` trong `kaggle-run.yml`; có `--first` để đánh số tiếp sau các session đã phát hành), rồi commit `experiments` và `kaggle-run.yml`.
- Chạy một session (trên Kaggle): `bash scripts/run_kaggle.sh <N>`; zip kết quả được đóng gói lại sau mỗi job.
- Nhận kết quả về máy: tải zip vào `runs_inbox/` (xem note KB `70_Van-hanh/kaggle-tai-ket-qua.md`; `node --env-file=<message-talisman>/.env <message-talisman>/bot/scripts/fetch-results.mjs <tag> --out runs_inbox`), rồi `python3 scripts/kaggle_queue.py ingest` (giải nén vào `outputs/`, idempotent) và `python3 src/summarize_runs.py outputs/<run>`.

## Quy tắc cứng
- **Giao thức dùng chung cho mọi phương pháp** (seed 42, 123, 2024, 7, 99; ViT5-base, AdamW lr 3e-4...). Đổi giao thức nghĩa là mọi phương pháp phải chạy lại: ghi vào `notes/DEVIATIONS.md`, không im lặng đổi.
- **Quy tắc dùng tập dữ liệu:** `train` dùng để thiết kế phương pháp; `dev` chỉ dùng **bên trong một lần chạy** (chọn checkpoint, early stopping), **không xem lỗi hay thành phần của dev để thiết kế augmentation**; `test` giải mã **một lần** với checkpoint chọn bằng dev, chỉ để báo cáo. Metric luôn tính lại từ `*_predictions.jsonl`.
- `data/restaurant_v1/` và các split đã làm sạch: **không sửa tay** (sinh bằng `scripts/prepare_data.py`).
- **Release tốn quota GPU của thành viên**: không tạo release hay chạy workflow `kaggle-run` khi chưa được hỏi. Workflow từ chối release có `sessions` khác `experiments/run_plan.json`, round đã `done`, hoặc tag không đúng tên round (tag là `<id>`, gửi lại là `<id>.<k>`, kiểm tra hạ tầng là `ci-smoke-<n>`). Pre-release chỉ là dry-run.
- **Mỗi release tối đa 12 session** (giới hạn của bot); số session lớn hơn phải chia thành nhiều release.
- Round cần `OPENAI_API_KEY` **không chạy được qua CI** (launcher không truyền secret Kaggle): chạy tay. **Không bao giờ gõ khóa vào ô lệnh của notebook**, không in `.env`. Mỗi script GPT bị chặn theo `OPENAI_MAX_USD` (mặc định 10 mỗi tháng, ghi ở `outputs/_openai_ledger.tsv`); đừng vòng qua giới hạn này.
- **Dữ liệu trong git:** sản phẩm cuối của augmentation (`data/aug/*.jsonl` kèm `.report.json`) được commit; các **file trung gian** (`data/aug/cv_mvpt/`, `pm_mvpt/`, `pa_ens/`, `pa_adat/`, `ac_ens/`, `*_inner/`, `data/subsets/*/f*/labels/`...) cố ý **chỉ nằm cục bộ** và nằm trong `.gitignore` (commit `105dc7f`), vì thực nghiệm không cần chúng. Đừng `git add -f` chúng, đừng xóa chúng mà chưa hỏi (chúng chỉ có một bản trên máy này).
- Không commit `.env`, zip kết quả (`results_*.zip`, `session_*.zip`, `runs_inbox/*.zip`), `INLG_2025_src.zip` và PDF paper của người khác (đã ignore hoặc nên chuyển sang Zotero).
- Không push, xóa kết quả hay tạo release khi chưa được hỏi.

## Điều cần biết
- Một session có job lỗi hoặc bị hoãn **thoát mã 1** (Kaggle `ERROR`, bot báo ❌) nhưng vẫn có zip kết quả của các job đã xong; không tự động thử lại: lập kế hoạch lại (`plan`) sau khi `ingest`.
- Mọi file mà job đọc phải **được commit trước khi publish release**, vì session chỉ thấy file đã nằm trong commit SHA (đã gặp: thiếu `data/aug/order_scores_train.json` làm hai job `cv_vi_mvpt` chết sau 0,3 phút, round `cv` ngày 16/11).
- Trên Kaggle mỗi GPU chạy một job một lúc (job dài nhất trước); job không còn vừa ngân sách thì bị bỏ qua và ở lại plan cho session sau.

## Liên kết
- Note dự án trong vault KB: `knowledge-base/20_Nghien-cuu/` (chưa có note chỉ mục riêng cho repo này)
- Quy trình chạy Kaggle CI chung: `knowledge-base/70_Van-hanh/kaggle-ci-pipeline.md`
- Sổ đăng ký project (cục bộ): `~/Desktop/Workspace/PROJECTS.md`
