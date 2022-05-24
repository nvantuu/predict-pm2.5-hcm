# Tải dữ liệu
## Tải dữ liệu khí tượng
- Dữ liệu khí tượng được tải từ trang [Dữ liệu khí tượng](https://weather.uwyo.edu/surface/meteorogram/seasia.shtml)
- Sử dụng GET request để truy vấn dữ liệu từ web server và lấy về máy tính
- File dùng để craw dữ liệu từ trang web trên `src/download_meteorology.js`. 
> Để có chạy được file này cần cài node version >= 16.15.0  

- Sau đó chạy lệnh sau (giả sử bạn đang ở folder cùng cấp với `src`)


```console
$ node src/download_meteorology_data.js
```

- Dữ liệu tải về sẽ được lưu trong folder data: bao gồm nhiều file csv mỗi file tương ứng dữ liệu khí
tượng của một ngày
## Tải dữ liệu PM2.5
[Dữ liệu PM2.5](https://www.airnow.gov/international/us-embassies-and-consulates/) của HCM được tải từ năm 2019 đến 2021.

# Tiền xử lý dữ liệu
Chi tiết quy trình, code tiền xử lý được mô tả rõ trong file `data/data_preprocessing.ipynb`
# Huấn luyện mô hình
Để huấn luyện mô hình LSTM-TSLightGBM kết hợp trước hết cần huấn luyện 2 mô hình đơn là LSTM

và LightGBM sau đó kết hợp chúng lại với trọng số là e1, e2. Quá trình này được gói gọn trong

module `main`.

```console
$ python3 src.main
```
# Test mô hình
Test hiệu quả dự đoán của mô hình LSTM-TSLightGBM trên tập test.

```console
$ python3 src.api_test --model_name
```
