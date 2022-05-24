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
# Huấn luyện mô hình
"# predict-pm2.5-hcm" 
"# predict-pm2.5-hcm" 
