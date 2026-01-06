import os
import time
import logging
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- HARICI KUTUPHANE ---
try:
    from ppg_lib import max30102
except ImportError:
    print("[HATA] 'ppg_lib' bulunamadi. ppg_lib klasorunu kontrol edin.")
    raise SystemExit(1)

# ==========================================
# SABITLER (KLINIK VE ISLEM AYARLARI)
# ==========================================
# Transfer-learning CNN RR modeli (10 uzunlukta RR sekansı, sigmoid çıkış)
MODEL_FILENAME = "cnn_rr_arrhythmia_transfer.h5"
AI_SEQUENCE_LEN = 10
VIEW_WINDOW = 400  # tampon boyutu (örnek)

# Sabit FS: MAX30102 tipik 100Hz
FS_TARGET = 100.0
DT = 1.0 / FS_TARGET

LOW_CUT = 0.5
HIGH_CUT = 8.0
FILTER_ORDER = 2

REFRACTORY_SEC = 0.35  # dicrotic notch'i atlamak için refrakter süre
MIN_RR_SECONDS = 0.30  # 200 BPM üzeri = fizyolojik dışı
MAX_RR_SECONDS = 1.80  # 33 BPM altı  = fizyolojik dışı

# Amplitüd SQI
AMP_MIN_FACTOR = 0.5   # %50 altı ise artefakt
AMP_MAX_FACTOR = 2.2   # %220 üstü ise artefakt
HARD_AMP_MAX = 12000.0 # saturasyon/clip için sert sınır
AMP_ROLL_LEN = 8

STATE_INIT = 0
STATE_LOCKED = 1
STATE_UNCERTAIN = 2
STATE_RESYNC = 3
MAX_MISSED_BEATS = 2

# BUFFERLAR
raw_buffer = collections.deque(maxlen=VIEW_WINDOW)
time_buffer = collections.deque(maxlen=VIEW_WINDOW)
rr_buffer = collections.deque(maxlen=AI_SEQUENCE_LEN)
amp_buffer = collections.deque(maxlen=AMP_ROLL_LEN)


# ==========================================
# YARDIMCI FONKSIYONLAR
# ==========================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def adaptive_elgendi_detector(signal_data, fs):
    """
    Elgendi PPG optimizasyonu:
      - Bandpass sonrası kare alma
      - İki hareketli ortalama (MA_peak ~120ms, MA_beat ~600ms) dinamik FS ile
      - Blok tespiti, lokal maksimum seçimi
      - AC genlik (peak - valley) ölçümü
    """
    sig = np.asarray(signal_data, dtype=float)
    squared = sig ** 2

    w1 = max(1, int(0.12 * fs))
    w2 = max(1, int(0.60 * fs))

    def moving_avg(x, win):
        kernel = np.ones(win) / win
        return np.convolve(x, kernel, mode="same")

    ma_peak = moving_avg(squared, w1)
    ma_beat = moving_avg(squared, w2)

    # Daha sağlam eşik: median tabanlı
    baseline = np.median(squared)
    threshold = ma_beat + (baseline * 0.05)
    blocks = ma_peak > threshold

    edges = np.diff(blocks.astype(int))
    starts = list(np.where(edges == 1)[0])
    ends = list(np.where(edges == -1)[0])
    if blocks[0]:
        starts.insert(0, 0)
    if blocks[-1]:
        ends.append(len(blocks) - 1)

    peaks = []
    for s, e in zip(starts, ends):
        if e - s < int(fs * 0.05):
            continue
        window = sig[s : e + 1]
        if window.size == 0:
            continue
        local_max_idx = int(np.argmax(window))
        peak_idx = s + local_max_idx

        # AC amplitude için peak öncesi 300ms içinde minimumu al
        valley_start = max(0, peak_idx - int(fs * 0.3))
        valley_window = sig[valley_start:peak_idx]
        if valley_window.size > 0:
            valley_amp = float(np.min(valley_window))
            peak_amp = float(sig[peak_idx])
            ac_amp = peak_amp - valley_amp
        else:
            ac_amp = 0.0
        peaks.append((peak_idx, ac_amp))

    return peaks


def streaming_peak_detector(filtered, fs, global_sample_index, last_peak_sample):
    """
    Event-based tepe tespiti.
    - Sadece son ~0.6s içindeki tek tepeyi arar.
    - Daha önce sayılmış peak'i tekrar vermez.
    """
    lookback = int(0.6 * fs)
    if len(filtered) < lookback + 2:
        return None

    segment = np.asarray(filtered[-lookback:], dtype=float)
    i = int(np.argmax(segment))

    # Kenarda tepe kabul etme
    if i == 0 or i == len(segment) - 1:
        return None

    # Minimum genişlik (40 ms) şartı
    min_width = max(1, int(0.04 * fs))
    if i < min_width or (len(segment) - i) < min_width:
        return None

    # Basit slope kontrolü (dicrotic/kenar gürültüsünü ele)
    if i < 2 or i > len(segment) - 3:
        return None
    left_slope = segment[i] - segment[i - 2]
    right_slope = segment[i] - segment[i + 2]
    if left_slope <= 0 or right_slope <= 0:
        return None

    peak_value = float(segment[i])
    seg_std = float(np.std(segment))
    if seg_std < 1e-6:
        return None

    # Basit amplitude eşiği: 0.3 * std
    if peak_value < 0.3 * seg_std:
        return None

    global_peak_sample = global_sample_index - (lookback - i)

    # Aynı peak'i tekrar sayma
    if global_peak_sample <= last_peak_sample:
        return None

    return global_peak_sample, peak_value


def is_amp_valid(amp):
    """Amplitüd SQI: saturasyon ve median tabanlı tutarlılık."""
    if amp <= 0 or amp > HARD_AMP_MAX:
        return False

    if len(amp_buffer) < AMP_ROLL_LEN:
        return True

    median_amp = float(np.median(amp_buffer))
    if median_amp <= 0:
        return True

    if amp < AMP_MIN_FACTOR * median_amp:
        return False
    if amp > AMP_MAX_FACTOR * median_amp:
        return False
    return True


def is_rr_consistent(rr, rr_buffer):
    """RR medianına göre %20 toleranslı tutarlılık kontrolü."""
    if len(rr_buffer) < 4:
        return True
    median_rr = float(np.median(rr_buffer))
    if median_rr <= 0:
        return True
    if rr < 0.8 * median_rr or rr > 1.2 * median_rr:
        return False
    return True


def zscore(signal_arr):
    mean = np.mean(signal_arr)
    std = np.std(signal_arr)
    if std < 1e-6:
        std = 1.0
    return (signal_arr - mean) / std


def load_model_or_exit():
    if not os.path.exists(MODEL_FILENAME):
        print(f"[HATA] Model dosyasi yok: {MODEL_FILENAME}")
        raise SystemExit(1)
    try:
        return tf.keras.models.load_model(MODEL_FILENAME, compile=False)
    except Exception as exc:
        print(f"[HATA] Model yuklenemedi: {exc}")
        raise SystemExit(1)


def init_sensor_or_exit():
    try:
        return max30102.MAX30102()
    except Exception as exc:
        print(f"[HATA] Sensor baslatilamadi: {exc}")
        raise SystemExit(1)


# ==========================================
# ANA PROGRAM
# ==========================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("arrhythmia")

    print("=" * 60)
    print("   CLINICAL-GRADE ARRHYTHMIA MONITOR (Raspberry Pi)")
    print("   Dynamic FS | Time-Synced RR | AF-Safe Logic")
    print("=" * 60)

    model = load_model_or_exit()
    sensor = init_sensor_or_exit()
    log.info("Model ve sensor baslatildi.")

    last_peak_sample = -1
    last_peak_time = 0.0
    sample_index = 0
    base_time = time.time()
    rr_state = STATE_INIT
    prev_rr_state = rr_state
    missed_beats = 0
    status_msg = "Sinyal bekleniyor..."
    status_color = "gray"

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    line, = ax.plot([], [], color="#00FF00", linewidth=1.5)
    ax.set_xlim(0, VIEW_WINDOW)
    ax.set_ylim(-3, 3)

    txt_bpm = ax.text(
        0.02, 0.92, "HR: --", transform=ax.transAxes,
        fontsize=16, color="white", weight="bold"
    )
    txt_rr = ax.text(
        0.20, 0.92, "RR: -- ms", transform=ax.transAxes,
        fontsize=12, color="gray"
    )
    txt_status = ax.text(
        0.02, 0.80, "Sinyal bekleniyor...", transform=ax.transAxes,
        fontsize=12, color="gray"
    )

    try:
        while True:
            available = sensor.available()
            if available <= 0:
                time.sleep(0.01)
                plt.pause(0.001)
                continue

            block_end = time.time()
            raw_chunk = []
            for _ in range(available):
                _, ir = sensor.read_fifo()
                if ir > 50000:  # temel kalite eşiği
                    raw_chunk.append(ir)

            if not raw_chunk:
                plt.pause(0.001)
                continue

            # --- Sabit FS ve monoton zaman ---
            times = []
            for _ in range(len(raw_chunk)):
                sample_index += 1
                t = base_time + sample_index * DT
                times.append(t)

            raw_buffer.extend(raw_chunk)
            time_buffer.extend(times)

            if len(raw_buffer) < VIEW_WINDOW:
                plt.pause(0.001)
                continue

            # --- Filtreleme ---
            sig = np.array(raw_buffer, dtype=float)
            filtered = butter_bandpass_filter(sig, LOW_CUT, HIGH_CUT, FS_TARGET, FILTER_ORDER)

            # --- Streaming tepe tespiti ---
            peak = streaming_peak_detector(filtered, FS_TARGET, sample_index, last_peak_sample)

            new_status_msg = status_msg
            new_status_color = status_color
            updated_status = False

            if peak:
                peak_sample, peak_amp = peak
                peak_time = base_time + peak_sample * DT

                if last_peak_sample >= 0:
                    rr_sec = (peak_sample - last_peak_sample) * DT

                    # -------------------------
                    # RR STATE MACHINE
                    # -------------------------
                    if rr_state == STATE_INIT:
                        last_peak_sample = peak_sample
                        last_peak_time = peak_time
                        rr_buffer.clear()
                        amp_buffer.clear()
                        missed_beats = 0
                        rr_state = STATE_LOCKED
                        log.info("STATE -> LOCKED (init)")
                        continue

                    if rr_state == STATE_LOCKED:
                        if rr_sec < REFRACTORY_SEC:
                            log.debug("Refrakter RR")
                            continue

                        if rr_sec > MAX_RR_SECONDS:
                            log.warning("RR cok buyuk (%.2fs) -> RESYNC", rr_sec)
                            rr_state = STATE_RESYNC
                            continue

                        if not (MIN_RR_SECONDS < rr_sec < MAX_RR_SECONDS):
                            missed_beats += 1
                            log.info("RR fizyolojik disi: %.2fs (%d)", rr_sec, missed_beats)
                            if missed_beats >= MAX_MISSED_BEATS:
                                rr_state = STATE_UNCERTAIN
                            continue

                        amp_abs = abs(peak_amp)
                        if not is_amp_valid(amp_abs):
                            missed_beats += 1
                            log.info("Amp reddedildi (%d)", missed_beats)
                            if missed_beats >= MAX_MISSED_BEATS:
                                rr_state = STATE_UNCERTAIN
                            continue

                        if not is_rr_consistent(rr_sec, rr_buffer):
                            log.info("RR outlier (soft): %.0f ms", rr_sec * 1000)
                            # Zaman ilerlemeli, senkron tutulmalı
                            last_peak_sample = peak_sample
                            last_peak_time = peak_time
                            missed_beats += 1
                            if missed_beats >= MAX_MISSED_BEATS:
                                rr_state = STATE_UNCERTAIN
                            continue

                        # --- RR KABUL ---
                        rr_buffer.append(rr_sec)
                        amp_buffer.append(amp_abs)
                        last_peak_sample = peak_sample
                        last_peak_time = peak_time
                        missed_beats = 0
                        txt_rr.set_text(f"RR: {rr_sec*1000:.0f} ms")
                        log.info("RR kabul: %.0f ms | amp=%.2f", rr_sec * 1000, amp_abs)

                        if len(rr_buffer) == AI_SEQUENCE_LEN:
                            arr = np.array(rr_buffer, dtype=float).reshape(1, AI_SEQUENCE_LEN, 1)
                            risk = float(model.predict(arr, verbose=0)[0][0])
                            bpm = int(60.0 / np.mean(rr_buffer))
                            txt_bpm.set_text(f"HR: {bpm}")

                            rr_std = float(np.std(rr_buffer))
                            rr_mean = float(np.mean(rr_buffer))
                            confidence = 0.0 if rr_mean <= 0 else max(0.0, 1.0 - (rr_std / rr_mean))

                            if confidence < 0.6:
                                new_status_msg = "Veri guvensiz (artefakt)"
                                new_status_color = "orange"
                                updated_status = True
                            else:
                                risk_adj = risk * confidence
                                log.info(
                                    "Inferans: HR=%d bpm | risk=%.3f | std=%.3f | conf=%.2f | risk_adj=%.3f",
                                    bpm, risk, rr_std, confidence, risk_adj
                                )

                                if risk_adj < 0.4:
                                    new_status_msg = f"Normal ritim ({risk_adj:.2f})"
                                    new_status_color = "#00FF00"
                                elif risk_adj > 0.6:
                                    new_status_msg = f"Aritmi olası ({risk_adj:.2f})"
                                    new_status_color = "#FF0000"
                                else:
                                    new_status_msg = f"Gri bölge / artefakt ({risk_adj:.2f})"
                                    new_status_color = "orange"
                                updated_status = True

                    elif rr_state == STATE_UNCERTAIN:
                        log.warning("STATE -> UNCERTAIN")
                        last_peak_sample = peak_sample
                        last_peak_time = peak_time
                        missed_beats = 0
                        if MIN_RR_SECONDS < rr_sec < MAX_RR_SECONDS:
                            rr_state = STATE_LOCKED
                            log.info("STATE -> LOCKED (recover)")

                    elif rr_state == STATE_RESYNC:
                        log.warning("STATE -> RESYNC")
                        rr_buffer.clear()
                        amp_buffer.clear()
                        last_peak_sample = peak_sample
                        last_peak_time = peak_time
                        missed_beats = 0
                        rr_state = STATE_LOCKED
                        txt_rr.set_text("RR: --")
                        txt_bpm.set_text("HR: --")

                else:
                    # İlk geçerli beat; sadece senkron başlatılır
                    last_peak_sample = peak_sample
                    last_peak_time = peak_time
                    rr_state = STATE_LOCKED
                    rr_buffer.clear()
                    amp_buffer.clear()
                    missed_beats = 0
                    log.info("STATE -> LOCKED (first beat)")

            # State değişti ama risk güncellenmediyse state tabanlı statüye geç
            if not updated_status and rr_state != prev_rr_state:
                if rr_state == STATE_LOCKED:
                    new_status_msg = "Ritim kilitli"
                    new_status_color = "#00FF00"
                elif rr_state == STATE_UNCERTAIN:
                    new_status_msg = "Ritim kararsiz"
                    new_status_color = "orange"
                else:
                    new_status_msg = "Senkron yenileniyor"
                    new_status_color = "#FF4444"

            # Statüyü sakla ve uygula
            status_msg = new_status_msg
            status_color = new_status_color
            prev_rr_state = rr_state

            # --- Görselleştirme (yalnızca normalize) ---
            norm_sig = zscore(filtered)
            line.set_ydata(norm_sig)
            line.set_xdata(np.arange(len(norm_sig)))
            line.set_color(status_color if peak else "#00FF00")

            txt_status.set_text(status_msg)
            txt_status.set_color(status_color)

            plt.draw()
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Kullanici tarafindan durduruldu.")
    finally:
        try:
            sensor.shutdown()
        except Exception:
            pass
        plt.close("all")


if __name__ == "__main__":
    main()