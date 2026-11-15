
from pathlib import Path
import subprocess
import tempfile

import cv2
import streamlit as st
import torch
import imageio_ffmpeg

from interpolate_video import load_model, infer_midpoint, postprocess


st.set_page_config(
    page_title="Video Frame Interpolation",
    page_icon="🎞️",
    layout="wide",
)

st.title("Video Frame Interpolation")
st.caption("U-Net · Vimeo-90K Triplet · PyTorch")

DEFAULT_CHECKPOINT = "checkpoints/best_model.pth"


@st.cache_resource
def get_model(checkpoint_path: str, device_name: str):
    device = torch.device(device_name)
    model = load_model(checkpoint_path, device)
    return model, device


def process_video(input_path, output_path, model, device, keep_fps,
                  progress_bar, status_text):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Không thể mở video đầu vào.")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Không đọc được kích thước video.")

    output_fps = source_fps if keep_fps else source_fps * 2.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV không thể tạo video đầu ra.")

    ok, previous = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise RuntimeError("Video không có frame đọc được.")

    writer.write(previous)
    pair_count = max(total_frames - 1, 1)
    processed_pairs = 0

    try:
        while True:
            ok, current = cap.read()
            if not ok:
                break

            midpoint = infer_midpoint(model, previous, current, device)
            midpoint_bgr = postprocess(midpoint, (width, height))

            # Original frame -> predicted midpoint -> next original frame
            writer.write(midpoint_bgr)
            writer.write(current)

            processed_pairs += 1
            previous = current

            progress_bar.progress(
                min(processed_pairs / pair_count, 1.0),
                text=f"Đang xử lý: {processed_pairs}/{pair_count} cặp frame",
            )
            status_text.caption(
                f"Đã xử lý {processed_pairs} cặp frame · "
                f"Đầu ra {output_fps:.2f} FPS"
            )
    finally:
        cap.release()
        writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Video tạm được tạo ra bị rỗng.")

    return {
        "width": width,
        "height": height,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "processed_pairs": processed_pairs,
    }


def convert_to_browser_mp4(source_path: Path, destination_path: Path):
    """Re-encode OpenCV's mp4v output to browser-friendly H.264 MP4."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i", str(source_path),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(destination_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr[-2500:] if result.stderr else "Không có log FFmpeg."
        raise RuntimeError("Không thể mã hóa video H.264:\n" + detail)

    if not destination_path.exists() or destination_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg không tạo được file MP4 H.264 hợp lệ.")


with st.sidebar:
    st.header("Cấu hình")
    checkpoint_path = st.text_input(
        "Checkpoint",
        value=DEFAULT_CHECKPOINT,
        help="Đường dẫn tới best_model.pth",
    )
    device_choice = st.selectbox(
        "Thiết bị xử lý",
        ["Tự động", "CUDA GPU", "CPU"],
        index=0,
    )
    keep_fps = st.checkbox(
        "Giữ nguyên FPS video gốc",
        value=False,
        help=(
            "Mặc định bỏ chọn: FPS đầu ra gấp đôi và thời lượng gần như giữ nguyên. "
            "Chọn mục này: FPS giữ nguyên và video dài khoảng gấp đôi."
        ),
    )

    if device_choice == "Tự động":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_choice == "CUDA GPU":
        device_name = "cuda"
    else:
        device_name = "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        st.error("PyTorch chưa nhận CUDA.")
    elif device_name == "cuda":
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("Đang chọn CPU.")


st.markdown(
    "Tải video lên, sau đó nhấn **Nội suy video**. Ứng dụng sẽ dự đoán "
    "một frame ở giữa mỗi cặp frame liên tiếp."
)

uploaded_file = st.file_uploader(
    "Chọn video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_file is not None:
    st.subheader("Video đầu vào")
    st.video(uploaded_file)
    st.caption(
        f"Tệp: {uploaded_file.name} · "
        f"Kích thước: {uploaded_file.size / (1024 * 1024):.2f} MB"
    )

    if st.button("Nội suy video", type="primary", use_container_width=True):
        checkpoint = Path(checkpoint_path)

        if not checkpoint.is_file():
            st.error(f"Không tìm thấy checkpoint: {checkpoint_path}")
        elif device_name == "cuda" and not torch.cuda.is_available():
            st.error("CUDA không khả dụng. Hãy kiểm tra PyTorch hoặc chọn CPU.")
        else:
            progress = st.progress(0, text="Đang chuẩn bị...")
            status = st.empty()

            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    safe_input_name = Path(uploaded_file.name).name
                    input_path = temp_dir / safe_input_name
                    raw_output = temp_dir / "interpolated_raw.mp4"
                    browser_output = temp_dir / "interpolated_h264.mp4"

                    input_path.write_bytes(uploaded_file.getvalue())

                    with st.spinner("Đang tải mô hình..."):
                        model, device = get_model(
                            str(checkpoint.resolve()),
                            device_name,
                        )

                    info = process_video(
                        input_path,
                        raw_output,
                        model,
                        device,
                        keep_fps,
                        progress,
                        status,
                    )

                    status.info("Đang chuyển video sang định dạng H.264...")
                    convert_to_browser_mp4(raw_output, browser_output)
                    output_bytes = browser_output.read_bytes()

                progress.progress(1.0, text="Hoàn tất")
                st.success("Đã nội suy và mã hóa video thành công.")

                left, right = st.columns(2)

                with left:
                    st.subheader("Video gốc")
                    st.video(uploaded_file)
                    st.caption(f"{info['source_fps']:.2f} FPS")

                with right:
                    st.subheader("Video sau nội suy")
                    st.video(output_bytes, format="video/mp4")
                    st.caption(f"{info['output_fps']:.2f} FPS · H.264")

                st.subheader("Thông tin xử lý")

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Độ phân giải",
                    f"{info['width']} × {info['height']}"
                )
                c2.metric(
                    "Cặp frame đã xử lý",
                    info["processed_pairs"]
                )
                c3.metric(
                    "FPS đầu ra",
                    f"{info['output_fps']:.2f}"
                )

                st.download_button(
                    "Tải video kết quả",
                    data=output_bytes,
                    file_name="interpolated_video.mp4",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True,
                )

                st.warning(
                    "Phiên bản này chưa giữ lại âm thanh của video gốc. "
                    "Nếu video dài, thời gian xử lý có thể lâu vì mô hình "
                    "phải dự đoán từng frame trung gian."
                )

            except Exception as exc:
                st.error(f"Lỗi khi xử lý video: {exc}")
                st.exception(exc)

else:
    st.info("Chưa có video. Hãy tải video lên để bắt đầu.")
