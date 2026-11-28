# Cách chạy

Giải nén file zip rồi vào thư mục vừa giải nén:

```bash
unzip asqp_llm_package.zip -d asqp_llm
cd asqp_llm
```

Kích hoạt env có sẵn torch (ví dụ `LLM`), trỏ tới thư mục chứa model, rồi chạy script setup:

```bash
conda activate LLM
export ASQP_CACHE_DIR=/duong/dan/toi/models
bash llm/setup.sh
```

Script sẽ tạo một venv riêng tên `.venv-asqp`, nên env `LLM` không bị đụng tới. Máy không có mạng cũng chạy được, vì thư viện đã có sẵn trong gói. Chạy xong nó in ra một loạt dòng kiểm tra; nếu tất cả đều là `PASS` thì đã cài xong.

Sau đó bật venv lên và chạy:

```bash
source .venv-asqp/bin/activate
nohup python llm/pipeline.py run > run_all.log 2>&1 &
```

Có 120 lần train: 8 nhánh × 3 bộ dữ liệu × 5 seed. Thứ tự chạy là baseline, phương pháp đề xuất, Copy trước, rồi đến 5 phương pháp literature (EDA, Concat, ADA, GenDA, RC). Như vậy, nếu phải dừng sớm, phần so sánh chính vẫn đã xong. Ước tính khoảng 175 giờ trên một A100 80GB (7–8 ngày); riêng nhánh đề xuất và Copy mất khoảng 100 giờ.

Muốn xem tiến độ thì `tail -f run_all.log`, hoặc `python llm/pipeline.py plan` để xem bước nào đã xong. Nếu bị ngắt giữa chừng, cứ chạy lại đúng lệnh trên, phần nào xong rồi sẽ tự bỏ qua. Muốn chạy một phần, ví dụ chỉ nhánh đề xuất trên Vi: `python llm/pipeline.py run --only vi_prop`.

Khi chạy xong sẽ có file `results_llm_*.zip` ở thư mục gốc, gửi lại file đó là được.
