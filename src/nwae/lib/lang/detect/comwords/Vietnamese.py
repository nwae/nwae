# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lib.lang.LangFeatures import LangFeatures
from nwae.lib.lang.detect.CommonWords import CommonWords
from nwae.lib.lang.nlp.LatinEquivalentForm import LatinEquivalentForm
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
import re


class Vietnamese(CommonWords):

    # We assume 20% as minimum
    MIN_VI_SENT_INTERSECTION_PCT = 0.2

    def __init__(
            self
    ):
        super().__init__(lang=LangFeatures.LANG_VN)

        # Taken from https://1000mostcommonwords.com/1000-most-common-vietnamese-words/
        self.raw_words = """
như
tôi
mình
mà
ông
là
cho
trên
là
với
họ
được
tại
một
có
này
từ
bởi
nóng
từ
nhưng
những gì
một số
là
nó
anh
hoặc
có
các
của
để
và
một
trong
chúng tôi
có thể
ra
khác
là
mà
làm
của họ
thời gian
nếu
sẽ
như thế nào
nói
một
môi
nói
không
bộ
ba
muốn
không khí
cũng
cũng
chơi
nhỏ
cuối
đặt
nhà
đọc
tay
cổng
lớn
chính tả
thêm
thậm chí
đất
ở đây
phải
lớn
cao
như vậy
theo
hành động
lý do tại sao
xin
người đàn ông
thay đổi
đi
ánh sáng
loại
tắt
cần
nhà
hình ảnh
thử
chúng tôi
một lần nữa
động vật
điểm
mẹ
thế giới
gần
xây dựng
tự
đất
cha
bất kỳ
mới
công việc
một phần
có
được
nơi
thực hiện
sống
nơi
sau khi
trở lại
ít
chỉ
chung quanh
người đàn ông
năm
đến
chương trình
mỗi
tốt
tôi
cung cấp cho
của chúng tôi
dưới
tên
rất
thông qua
chỉ
hình thức
câu
tuyệt vời
nghi
nói
giúp
thấp
dòng
khác nhau
lần lượt
nguyên nhân
nhiều
có nghĩa là
trước
di chuyển
ngay
cậu bé
cũ
quá
như nhau
cô
tất cả
có
khi
lên
sử dụng
của bạn
cách
về
nhiều
sau đó
họ
viết
sẽ
như
để
các
cô
lâu
làm
điều
thấy
anh
hai
có
xem
hơn
ngày
có thể
đi
đến
đã làm
số
âm thanh
không có
nhất
nhân dân
của tôi
hơn
biết
nước
hơn
gọi
đầu tiên
người
có thể
xuống
bên
được
bây giờ
tìm
đầu
đứng
riêng
trang
nên
nước
tìm thấy
câu trả lời
trường
phát triển
nghiên cứu
vẫn
học
nhà máy
bìa
thực phẩm
ánh nắng mặt trời
bốn
giữa
nhà nước
giữ
mắt
không bao giờ
cuối cùng
cho phép
nghĩ
thành phố
cây
qua
trang trại
cứng
bắt đầu
might
câu chuyện
cưa
đến nay
biển
vẽ
còn lại
cuối
chạy
không
trong khi
báo chí
gần
đêm
thực
cuộc sống
số
phía bắc
cuốn sách
thực hiện
mất
khoa học
ăn
phòng
người bạn
bắt đầu
ý tưởng
cá
núi
ngăn chặn
một lần
cơ sở
nghe
ngựa
cắt
chắc chắn
xem
màu
khuôn mặt
gỗ
chính
mở
dường như
cùng
tiếp theo
trắng
trẻ em
bắt đầu
có
đi bộ
Ví dụ
giảm bớt
giấy
nhóm
luôn luôn
nhạc
những
cả hai
đánh dấu
thường
thư
cho đến khi
dặm
sông
xe
chân
chăm sóc
thứ hai
đủ
đồng bằng
cô gái
thông thường
trẻ
sẵn sàng
trên đây
bao giờ
màu đỏ
danh sách
mặc dù
cảm thấy
nói chuyện
chim
sớm
cơ thể
con chó
gia đình
trực tiếp
đặt ra
lại
bài hát
đo lường
cửa
sản phẩm
đen
ngắn
chữ số
lớp
gió
câu hỏi
xảy ra
hoàn thành
tàu
khu vực
một nửa
đá
để
lửa
nam
vấn đề
mảnh
nói
biết
vượt qua
từ
đầu
toàn bộ
vua
đường phố
inch
nhân
không có gì
Tất nhiên
ở lại
bánh xe
đầy đủ
lực
màu xanh
đối tượng
quyết định
bề mặt
sâu
mặt trăng
đảo
chân
hệ thống
bận rộn
kiểm tra
ghi
thuyền
phổ biến
vàng
có thể
máy bay
thay
khô
tự hỏi
cười
ngàn
trước
ran
kiểm tra
trò chơi
hình dạng
đánh đồng
nóng
bỏ lỡ
mang
nhiệt
tuyết
lốp xe
mang lại
vâng
xa
điền
đông
sơn
ngôn ngữ
trong
đơn vị
điện
thị trấn
tốt
nhất định
bay
giảm
dẫn
kêu
tối
máy
ghi
đợi
kế hoạch
con số
sao
hộp
danh từ
lĩnh vực
phần còn lại
chính xác
thể
bảng
Xong
vẻ đẹp
ổ đĩa
đứng
chứa
trước
dạy
tuần
thức
đã
màu xanh lá cây
oh
nhanh chóng
phát triển
đại dương
ấm áp
miễn phí
phút
mạnh mẽ
đặc biệt
tâm
sau
trong
đuôi
sản xuất
thực tế
không gian
nghe
tốt nhất
giờ
tốt hơn
đúng
trong khi
trăm
năm
nhớ
bước
đầu
giư
tây
mặt đất
quan tâm
đạt
nhanh chóng
động từ
hát
lắng nghe
sáu
bảng
du lịch
ít
buổi sáng
mười
đơn giản
nhiều
nguyên âm
hướng
chiến tranh
đặt
chống lại
mô hình
chậm
trung tâm
tình yêu
người
tiền
phục vụ
xuất hiện
đường
Bản đồ
mưa
quy tắc
phối
kéo
lạnh
thông báo
giọng nói
năng lượng
săn
có thể xảy ra
giường
anh trai
trứng
đi xe
pin
tin
có lẽ
chọn
đột ngột
tính
vuông
lý do
chiều dài
đại diện
nghệ thuật
Tiêu đề
khu
kích thước
khác nhau
giải quyết
nói
trọng lượng
chung
băng
vấn đề
vòng tròn
đôi
bao gồm
chia
âm tiết
cảm thấy
lớn
bóng
nhưng
sóng
rơi
tim
là
hiện nay
nặng
khiêu vũ
động cơ
vị trí
cánh tay
rộng
buồm
tài liệu
phần
rừng
ngồi
cuộc đua
cửa sổ
cửa hàng
mùa hè
đào tạo
ngủ
chứng minh
đơn độc
chân
tập thể dục
tường
bắt
mount
muốn
bầu trời
hội đồng quản trị
niềm vui
mùa đông
ngồi
bằng văn bản
hoang dã
cụ
giữ
kính
cỏ
bò
công việc
cạnh
dấu hiệu
lần
qua
mềm
vui vẻ
sáng
khí
thời tiết
tháng
triệu
chịu
kết thúc
hạnh phúc
hy vọng
hoa
mặc
lạ
ra đi
thương mại
giai điệu
chuyến đi
văn phòng
nhận
hàng
miệng
chính xác
biểu tượng
chết
nhất
rắc rối
hét lên
trừ
đã viết
hạt giống
giai điệu
tham gia
đề nghị
sạch
nghỉ
phụ nữ
sân
tăng
xấu
đòn
dầu
máu
chạm
tăng
phần trăm
trộn
đội
dây
chi phí
thua
nâu
mặc
vườn
như nhau
gửi
chọn
giảm
phù hợp với
chảy
công bằng
ngân hàng
thu thập
lưu
kiểm soát
số thập phân
tai
khác
khá
đã phá vỡ
khi
trung
giết
con trai
hồ
thời điểm
quy mô
lớn
mùa xuân
quan sát
con
thẳng
phụ âm
quốc gia
từ điển
sưa
tốc độ
phương pháp
cơ quan
trả
tuổi
phần
váy
điện toán đám mây
bất ngờ
yên tĩnh
đá
nhỏ
lên cao
mát mẻ
thiết kế
người nghèo
rất nhiều
thí nghiệm
dưới
chính
sắt
đơn
thanh
phẳng
hai mươi
da
nụ cười
nếp
lỗ
nhảy
bé
tám
làng
đáp ứng
gốc
mua
nâng cao
giải quyết
kim loại
liệu
đẩy
bảy
đoạn
thứ ba
có trách nhiệm
được tổ chức
lông
mô tả
nấu ăn
sàn
hoặc
kết quả
ghi
đồi
an toàn
mèo
thế kỷ
xem xét
loại
pháp luật
bit
bờ biển
bản sao
cụm từ
im lặng
cao
cát
đất
cuộn
nhiệt độ
ngón tay
ngành công nghiệp
giá trị
cuộc chiến
lời nói dối
đánh bại
kích thích
tự nhiên
xem
ý nghĩa
vốn
sẽ không
ghế
nguy hiểm
trái cây
giàu
dày
người lính
quá trình
hoạt động
thực hành
riêng biệt
khó khăn
bác sĩ
xin vui lòng
bảo vệ
trưa
cây trồng
hiện đại
yếu tố
nhấn
sinh viên
góc
bên
cung cấp
có
xác định vị trí
vòng
nhân vật
côn trùng
bắt
thời gian
chỉ ra
radio
nói
nguyên tử
con người
lịch sử
hiệu lực
điện
mong đợi
xương
đường sắt
tưởng tượng
cho
đồng ý
do đó
nhẹ nhàng
người phụ nữ
đội trưởng
đoán
cần thiết
sắc nét
cánh
tạo
hàng xóm
rửa
bat
thay
đám đông
ngô
so sánh
bài thơ
chuỗi
chuông
phụ thuộc
thịt
chà
ống
nổi tiếng
đồng đô la
sông
sợ hãi
cảnh
mỏng
tam giác
hành tinh
nhanh
trưởng
thuộc địa
đồng hồ
tôi
cà vạt
nhập
chính
tươi
tìm kiếm
gửi
vàng
súng
cho phép
in
chết
tại chỗ
sa mạc
phù hợp với
hiện tại
thang máy
tăng
đến
chủ
theo dõi
mẹ
bờ
phân chia
tờ
chất
ủng hộ
kết nối
bài
chi tiêu
hợp âm
chất béo
vui
ban đầu
chia sẻ
trạm
cha
bánh mì
phí
thích hợp
thanh
phục vụ
phân khúc
nô lệ
vịt
ngay lập tức
thị trường
mức độ
cư
gà
thân yêu
kẻ thù
trả lời
ly
xảy ra
hỗ trợ
bài phát biểu
thiên nhiên
phạm vi
hơi nước
chuyển động
con đường
chất lỏng
đăng nhập
có nghĩa là
thương
răng
vỏ
cổ
oxy
đường
chết
khá
kỹ năng
phụ nữ
mùa
giải pháp
nam châm
bạc
cảm ơn
chi nhánh
trận đấu
hậu tố
đặc biệt là
sung
sợ
to
em gái
thép
thảo luận
về phía trước
tương tự
hướng dẫn
kinh nghiệm
điểm
táo
mua
dẫn
sân
áo
khối lượng
thẻ
ban nhạc
dây
trượt
giành chiến thắng
mơ
buổi tối
điều kiện
thức ăn chăn nuôi
công cụ
tổng số
cơ bản
mùi
thung lũng
cũng không
đôi
ghế
tiếp tục
khối
biểu đồ
mũ
bán
thành công
công ty
trừ
sự kiện
riêng
thỏa thuận
bơi
hạn
ngược lại
vợ
giày
vai
lây lan
sắp xếp
trại
phát minh
bông
Sinh
xác định
lít
chín
xe tải
tiếng ồn
mức
cơ hội
thu thập
cửa hàng
căng ra
ném
tỏa sáng
tài sản
cột
phân tử
chọn
sai
màu xám
lặp lại
yêu cầu
rộng
chuẩn bị
muối
mui
số nhiều
tức giận
xin
lục
"""
        word_split_token = BasicPreprocessor.get_word_separator(
            lang = LangFeatures.LANG_VI
        )
        self.raw_words = re.sub(
            pattern = '[\n\r\t]',
            repl    = word_split_token,
            string  = self.raw_words
        )
        #
        # For Vietnamese we also need to convert to latin equivalent form
        #
        latin_equival_form = LatinEquivalentForm.get_latin_equivalent_form_vietnamese(
            word = self.raw_words
        )
        self.raw_words = self.raw_words + word_split_token + latin_equival_form

        # We are not splitting the
        self.process_common_words(
            word_split_token = word_split_token
        )

        return

    def get_min_threshold_intersection_pct(
            self
    ):
        return Vietnamese.MIN_VI_SENT_INTERSECTION_PCT

if __name__ == '__main__':
    obj = Vietnamese()
    print(obj.common_words)
    exit(0)
