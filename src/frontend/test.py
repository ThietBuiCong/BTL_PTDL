import streamlit as st

# Tiêu đề trang
st.title("Chào mừng đến với ứng dụng Streamlit đầu tiên! 🎉")

# Viết chữ / văn bản
st.write("Đây là một ứng dụng web mini được viết hoàn toàn bằng Python.")

# Tạo ô nhập liệu cho người dùng
name = st.text_input("Nhập tên của bạn:")

if name:
    st.success(f"Xin chào **{name}**! Chúc bạn một ngày tốt lành.")

# Tạo nút bấm tương tác
if st.button("Bấm vào đây đi"):
    st.balloons()  # Hiệu ứng bóng bay chúc mừng trên màn hình
    st.write("Bạn vừa kích hoạt hiệu ứng thành công! 🚀")

# Tạo slider chọn số
number = st.slider("Chọn một số bất kỳ:", min_value = 10)
st.write(f"Số bạn đang chọn là: **{number}**")