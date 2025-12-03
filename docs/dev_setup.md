# Thiết lập môi trường dev

Để VS Code/Pylance không báo lỗi `Import "pytest" could not be resolved`, hãy dùng virtualenv và cài đủ phụ thuộc dev.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows dùng .venv\\Scripts\\activate
pip install -r requirements-dev.txt
```

Pyright/Pylance sẽ tìm gói trong `.venv` khi dùng `pyrightconfig.json` đi kèm repo. Nếu bạn dùng venv khác, cập nhật lại `venv`/`venvPath` trong file cấu hình này cho phù hợp.
