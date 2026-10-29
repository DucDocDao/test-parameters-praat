import numpy as np
import parselmouth
import sys
from pathlib import Path


def extract_features_praat(audio_path, verbose=False):
    snd = parselmouth.Sound(str(audio_path))

    # =========================================================================
    # TỰ CHỌN GIAI ĐOẠN BẰNG CÁCH BẬT/TẮT COMMENT (#) Ở 1 TRONG 2 ĐOẠN DƯỚI ĐÂY
    # =========================================================================

    # --- GIAI ĐOẠN 1: Không truyền tham số (Mặc định) ---
    # pitch_object = snd.to_pitch()

    # --- GIAI ĐOẠN 2: Tự truyền tham số thủ công ---
    # (Nếu chỉ test 3 tham số cơ bản thì dùng snd.to_pitch, nếu test đủ 10 tham số thì dùng snd.to_pitch_ac)
    pitch_object = snd.to_pitch_ac(
         time_step=0.01,
         pitch_floor=75.0,
         max_number_of_candidates=15,
         very_accurate=False,
         silence_threshold=0.03,
         voicing_threshold=0.45,
         octave_cost=0.01,
         octave_jump_cost=0.35,
         voiced_unvoiced_cost=0.14,
         pitch_ceiling=600.0
     )

    # =========================================================================

    pitch = np.nan_to_num(pitch_object.selected_array['frequency'])
    strength = np.nan_to_num(pitch_object.selected_array['strength'])
    time_stamps = pitch_object.xs()

    if verbose:
        print(f"\n[KIỂM CHỨNG BƯỚC NHẢY] Các mốc thời gian của {audio_path} - {len(pitch)} cao độ:\n{time_stamps[:len(pitch)]}\n")

    return pitch, time_stamps, strength


def compute_metrics(pitch, time_stamps, strength):
    total_frames = len(pitch)
    voiced_mask = pitch > 0
    voiced_frames = int(np.sum(voiced_mask))
    voiced_ratio = (voiced_frames / total_frames * 100) if total_frames > 0 else 0.0

    # Cao độ và độ mạnh trung bình (chỉ tính trên các frame có tiếng)
    mean_pitch = float(np.mean(pitch[voiced_mask])) if voiced_frames > 0 else 0.0
    median_pitch = float(np.median(pitch[voiced_mask])) if voiced_frames > 0 else 0.0
    mean_strength = float(np.mean(strength[voiced_mask])) if voiced_frames > 0 else 0.0

    # Số lần chuyển trạng thái Hữu thanh <-> Vô thanh (độ đứt quãng)
    transitions = int(np.sum(np.diff(voiced_mask) != 0)) if total_frames > 1 else 0

    # Số lần nhảy quãng tám đột ngột giữa 2 frame liên tiếp cùng có tiếng
    both_voiced = voiced_mask[:-1] & voiced_mask[1:]
    if np.any(both_voiced):
        log2_ratios = np.abs(np.log2(pitch[1:][both_voiced] / pitch[:-1][both_voiced]))
        octave_jumps = int(np.sum(log2_ratios >= 0.8))
    else:
        octave_jumps = 0

    return {
        'frames': total_frames,
        'start_t': float(time_stamps[0]) if total_frames > 0 else 0.0,
        'end_t': float(time_stamps[-1]) if total_frames > 0 else 0.0,
        'voiced_frames': voiced_frames,
        'voiced_ratio': voiced_ratio,
        'mean_pitch': mean_pitch,
        'median_pitch': median_pitch,
        'mean_strength': mean_strength,
        'transitions': transitions,
        'octave_jumps': octave_jumps,
    }


def main():
    audio_dir = Path(__file__).parent / "audio"
    audio_files = sorted(audio_dir.rglob("*.wav"))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not audio_files:
        print(f"Không tìm thấy file .wav nào trong thư mục: {audio_dir}")
        return

    print("=" * 110)
    print(f"ĐANG XỬ LÝ {len(audio_files)} FILE TRONG THƯ MỤC: {audio_dir.name}/")
    print("=" * 110)
    print(f"{'STT':<5} | {'Tên File':<34} | {'Frames':<7} | {'Voiced (%)':<11} | {'Mean Hz':<9} | {'Strength':<8} | {'Ngắt':<5} | {'Nhảy Q8'}")
    print("-" * 110)

    all_stats = []

    for idx, file_path in enumerate(audio_files, 1):
        pitch, time_stamps, strength = extract_features_praat(file_path)
        m = compute_metrics(pitch, time_stamps, strength)
        all_stats.append(m)

        # In từng dòng gọn gàng cho mỗi file
        name = str(file_path.relative_to(audio_dir))
        name = (name[:31] + "...") if len(name) > 34 else name
        print(f"{idx:<5} | {name:<34} | {m['frames']:<7} | {m['voiced_ratio']:>9.2f}% | {m['mean_pitch']:>9.2f} | {m['mean_strength']:>8.4f} | {m['transitions']:<5} | {m['octave_jumps']}")

    # BẢNG TỔNG KẾT ĐẦY ĐỦ CHO CẢ 1000 AUDIO CỦA LẦN CHẠY HIỆN TẠI
    avg = lambda key: np.mean([x[key] for x in all_stats])
    total = lambda key: np.sum([x[key] for x in all_stats])

    print("=" * 110)
    print(f"TỔNG KẾT OUTPUT CỦA {len(all_stats)} AUDIO:")
    print("-" * 110)
    print(f"  1. Tổng số khung (Total Frames)       : {total('frames')} frames (Trung bình: {avg('frames'):.2f} frames/file)")
    print(f"  2. Mốc thời gian TB (Đầu -> Cuối)     : {avg('start_t'):.4f}s -> {avg('end_t'):.4f}s")
    print(f"  3. Tỷ lệ hữu thanh TB (Voiced Ratio)  : {avg('voiced_ratio'):.2f}% (Tổng: {total('voiced_frames')} voiced frames)")
    print(f"  4. Cao độ trung bình (Mean Pitch)     : {avg('mean_pitch'):.2f} Hz")
    print(f"  5. Cao độ trung vị (Median Pitch)     : {avg('median_pitch'):.2f} Hz")
    print(f"  6. Độ mạnh/tự tin TB (Mean Strength)  : {avg('mean_strength'):.4f}")
    print(f"  7. Tổng số lần ngắt tiếng (Transitions): {total('transitions')} lần (Trung bình: {avg('transitions'):.2f} lần/file)")
    print(f"  8. Tổng số lần nhảy quãng 8 (Jumps)   : {total('octave_jumps')} lần (Trung bình: {avg('octave_jumps'):.2f} lần/file)")
    print("=" * 110)


if __name__ == "__main__":
    main()