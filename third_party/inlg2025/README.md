# Notebook gốc của INLG 2025 (ASQP dạng sinh code)

Hai notebook của nhóm tác giả bài INLG 2025 ("ABSQA as code generation"), lấy từ `INLG_2025_src.zip`:

| File | Nội dung |
|---|---|
| `LLM_Gemma.ipynb` | Gemma-7B + LoRA, prompt `coding_format`, train và decode trên Vi-AbSQA |
| `LLM_QWen.ipynb` | Qwen2.5-14B-Instruct + LoRA, prompt `json_format` |

Output của các cell đã được xóa trước khi commit, vì chúng chứa đường dẫn tài khoản trên server; code giữ nguyên.

## Những gì không có ở đây

- Các module mà notebook import (`eval_utils`, `prompting`, `prompting_qwen`, `utils`) **không có trong archive gốc**, nên hai notebook không chạy được một mình.
- Dữ liệu notebook đọc (`dataset/viabsa_rest/csv/{train,dev,test}.csv`) cũng không có. Bản Vi-AbSQA đã làm sạch của repo nằm ở `data/restaurant_v1/`.
- Archive gốc còn có `train2.csv`, `test_2.csv` và `finetune_laya(1).ipynb`, thuộc một bài toán khác (gán nhãn linh kiện), không liên quan tới ASQP. Ba file này cố ý không đưa lên vì repo là public. Archive vẫn nằm trong `.gitignore`.

## Muốn chạy công thức này thì dùng `llm/`

`llm/inlg.py` cài lại đúng công thức của hai notebook (model, LoRA, lr, số epoch, NEFTune, beam 5, prompt). Những chỗ khác với notebook được liệt kê trong docstring đầu file, và không chỗ nào làm thay đổi công thức. Package `llm/` chạy được offline, trên đúng các bộ dữ liệu của bảng kết quả chính. Cách setup và chạy: xem [`llm/README.md`](../../llm/README.md).
