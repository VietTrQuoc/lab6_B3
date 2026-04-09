# Hướng dẫn chạy project Lab6

## 1. Cài biến môi trường (Windows)

Mở **Terminal (PowerShell)** và chạy:

``` powershell
setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxx"
```

> ⚠️ Sau khi chạy lệnh này, hãy **đóng và mở lại terminal** để biến môi
> trường có hiệu lực.

------------------------------------------------------------------------

## 2. Kích hoạt môi trường ảo (venv)

Di chuyển vào thư mục project và chạy:

``` powershell
venv\Scripts\activate
```

------------------------------------------------------------------------

## 3. Cài thư viện (nếu cần)

Nếu project có file `requirements.txt`, chạy:

``` powershell
pip install -r requirements.txt
```

> Nếu thư viện đã được cài trước đó thì có thể bỏ qua bước này.

------------------------------------------------------------------------

## 4. Chạy ứng dụng

Di chuyển vào folder **Lab6**, sau đó chạy:

``` powershell
uvicorn app.main:app --reload
```

Nếu thành công, server sẽ chạy ở địa chỉ mặc định:

    http://127.0.0.1:8000

------------------------------------------------------------------------

