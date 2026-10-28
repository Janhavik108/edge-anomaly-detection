#include <Arduino.h>
#include <driver/i2s.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "presigmoid_model_data.h"  // <-- rename to match your actual .h filename

// --- Hardware Pin Definitions ---
#define I2S_WS          25
#define I2S_SD          32
#define I2S_SCK         33
#define LED_ALERT_PIN   2

// --- Spectrogram Pipeline Parameters ---
#define SAMPLE_RATE     16000
#define N_FFT           512
#define HOP_LENGTH      256
#define N_MELS          32
// ---------------------------------------------------------------------------
// UPDATED: the retrained model uses a 1.0s window (64 STFT frames at
// hop=256) instead of the original 500ms/32-frame window -- the shorter
// window couldn't reliably capture slow, once-per-revolution effects like
// imbalance/wobble at low fan speeds. Input tensor shape is now
// (N_MELS=32, TIME_FRAMES=64, 1), matching Input(shape=(N_MELS,
// TIME_FRAMES, 1)) in the retrained notebook. Requires the new
// anomaly_model_data.h exported from that notebook's Cell 5 -- the old
// 32x32 model will not load correctly against this capture loop.
// ---------------------------------------------------------------------------
#define TIME_FRAMES     64
#define TOP_DB          80.0f

// ---------------------------------------------------------------------------
// UPDATED: two-sided threshold, replacing the single ANOMALY_THRESHOLD.
// Per-fault-type analysis on the training data showed two OPPOSITE failure
// signatures: bearing rub / blade strike / tamper all INCREASE reconstruction
// error (as expected), but obstruction and fast-speed-variation DECREASE it
// below the normal baseline -- a single "MSE > threshold" check scored those
// two fault types worse than random (one-sided AUC 0.09 and 0.21), while a
// two-sided check caught them well (0.87 and 0.79).
//
// *** THESE ARE PLACEHOLDERS -- DO NOT DEPLOY WITH THESE VALUES. ***
// Python-side MSE numbers do NOT transfer directly to firmware numbers (real
// mic noise, room acoustics, and INT8 quantization shift the whole scale --
// we've seen this repeatedly in this project). Calibrate for real:
//   1. Flash this firmware with the new model, let it run past warmup.
//   2. Let it run for 2-3 minutes under known-NORMAL conditions only.
//   3. From the Serial log, compute mean and std of the "Smoothed" values.
//   4. Set: ANOMALY_THRESHOLD_LOW  = mean - (K_LOW  * std)
//           ANOMALY_THRESHOLD_HIGH = mean + (K_HIGH * std)
//      K=3 is a reasonable starting point for both; widen the side that's
//      false-alarming, narrow the side that's missing real faults.
// Paste that Serial log back and I'll compute the real numbers with you.
// ---------------------------------------------------------------------------
float ANOMALY_THRESHOLD_LOW  = 0.00100f;  // PLACEHOLDER -- calibrate on-device
float ANOMALY_THRESHOLD_HIGH = 0.00300f;  // PLACEHOLDER -- calibrate on-device
const int CONSECUTIVE_ANOMALIES_REQUIRED = 2;
int anomaly_streak = 0;

// ---------------------------------------------------------------------------
// ADDED: the I2S peripheral and the INMP441 itself need a short settling
// period right after boot -- the first several captures can contain
// transient garbage that isn't representative of steady-state audio. This
// showed up as a burst of false FAULT readings immediately after every
// reset that decayed back to baseline over ~15-40 readings. Discard the
// first WARMUP_CYCLES captures entirely (don't push them into the
// smoothing history or evaluate them against the threshold).
// UPDATED: each cycle now captures ~1.0s of audio instead of ~0.5s, so the
// cycle count was halved (20 -> 10) to keep roughly the same wall-clock
// warmup duration rather than doubling it.
// ---------------------------------------------------------------------------
#define WARMUP_CYCLES 10
int cycle_count = 0;

// ---------------------------------------------------------------------------
// ADDED: Smooth MSE over a short rolling window before comparing to the
// threshold. Single 274ms frames are noisy enough to cross the threshold in
// either direction on their own -- a real fault is a sustained change, not a
// ---------------------------------------------------------------------------
// EXPONENTIAL MOVING AVERAGE instead of a flat window average. A flat N-frame
// average takes N full cycles to catch up to a step change in EITHER
// direction (rising into a fault or falling back to normal) -- with N=8 and
// ~819ms/cycle that's ~6.5s of lag each way, which is what caused the slow
// detection. EMA reacts to a sudden change on the very next cycle instead of
// waiting for old samples to age out, while still rejecting single-frame
// noise. EMA_ALPHA is the only knob: higher = faster reaction, more jittery;
// lower = smoother, slower. 0.35 roughly halves the reaction lag vs the old
// 8-frame average -- raise toward 0.5-0.6 for even faster response, or lower
// toward 0.15-0.2 if it starts reacting to single noisy frames.
// ---------------------------------------------------------------------------
#define EMA_ALPHA 0.35f
float smoothed_mse_ema = 0.0f;
bool ema_initialized = false;

float pushAndGetSmoothedMSE(float new_mse) {
    if (!ema_initialized) {
        smoothed_mse_ema = new_mse;  // seed with first real reading, not 0
        ema_initialized = true;
    } else {
        smoothed_mse_ema = EMA_ALPHA * new_mse + (1.0f - EMA_ALPHA) * smoothed_mse_ema;
    }
    return smoothed_mse_ema;
}

// --- TFLM Tensor Arena ---
// UPDATED: this board has NO PSRAM (confirmed via boot log: "PSRAM chip not
// found"), and its largest contiguous internal-heap block was measured at
// ~108-110KB, stable across boots. That's a real ceiling, not a fluke --
// don't request more than that. The original wide model needed ~164KB
// (167952 bytes) and could never fit here regardless of tuning; the
// narrowed-channel architecture in Cell 3 is meant to bring that down.
// Starting at 100KB, deliberately under the ceiling, so a failure here
// reports the model's REAL requirement via AllocateTensors()'s error
// message rather than failing on the malloc itself. Update this constant
// once you know that real number.
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t* tensor_arena = nullptr;

// Custom Robust Error Reporter
class RobustSerialReporter : public tflite::ErrorReporter {
public:
    int Report(const char* format, va_list args) override {
        char buffer[256];
        vsnprintf(buffer, sizeof(buffer), format, args);
        Serial.print("[TFLM] ");
        Serial.print(buffer);
        return 0;
    }
};

static RobustSerialReporter serial_reporter;
static tflite::MicroMutableOpResolver<20> resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Compact Working Buffers
int32_t i2s_raw_buffer[HOP_LENGTH];
float sliding_window[N_FFT];
float fft_real[N_FFT];
float fft_imag[N_FFT];
float power_spectrum[N_FFT / 2 + 1];
// ---------------------------------------------------------------------------
// FIXED: moved from a local (stack) variable inside captureLibrosaSpectrogram()
// to file scope. At TIME_FRAMES=64, this array is 64*32*4 = 8192 bytes -- on
// its own, roughly the entire default loopTask stack budget (commonly 8KB on
// Arduino-ESP32), which caused a stack canary overflow crash right after the
// first capture call. Matches the existing pattern for sliding_window /
// fft_real / fft_imag / power_spectrum above, all of which are file-scope
// for the same reason.
// ---------------------------------------------------------------------------
float frame_mels[TIME_FRAMES][N_MELS];

// ---------------------------------------------------------------------------
// FIXED: These bin boundaries are recomputed for the ACTUAL config in use
// (N_FFT=512, SAMPLE_RATE=16000, N_MELS=32 -> valid bin range is 0..256).
// The previous table contained values up to 461, which is past the end of
// power_spectrum[257] and caused an out-of-bounds stack read on every frame,
// corrupting roughly the top half of the mel bins with garbage data. That's
// why MSE sat flat and high (~0.17-0.21) regardless of actual fan condition.
// This table matches librosa's default (htk=False) mel scale for the same
// sr/n_fft/n_mels used in training.
// ---------------------------------------------------------------------------
const int mel_bin_indices[N_MELS + 2] = {
    0, 3, 6, 9, 12, 15, 18, 20, 23, 26, 29, 32, 35, 39, 43, 47,
    52, 57, 62, 68, 75, 83, 91, 100, 110, 120, 132, 145, 160, 176, 193, 212, 233, 256
};

// ---------------------------------------------------------------------------
// FIXED: Slaney-style area normalization per mel filter, matching librosa's
// default (norm='slaney'). Without this, wider high-frequency filters get
// over-weighted relative to what the model was trained on -- a shape
// distortion that survives the ref=max normalization below. Computed as
// 2 / (mel_freq[m+2] - mel_freq[m]) for sr=16000, n_fft=512, n_mels=32.
// ---------------------------------------------------------------------------
const float mel_norm[N_MELS] = {
    0.01094028f, 0.01094028f, 0.01094028f, 0.01094028f, 0.01094028f, 0.01094028f,
    0.01094028f, 0.01094028f, 0.01094028f, 0.01092912f, 0.01047121f, 0.00958577f,
    0.00872346f, 0.00793872f, 0.00722457f, 0.00657466f, 0.00598322f, 0.00544498f,
    0.00495517f, 0.00450941f, 0.00410375f, 0.00373459f, 0.00339863f, 0.00309290f,
    0.00281467f, 0.00256147f, 0.00233105f, 0.00212135f, 0.00193052f, 0.00175685f,
    0.00159881f, 0.00145499f
};

void setupI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 6,
        .dma_buf_len = 128,
        .use_apll = false
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
}

// In-place Radix-2 Cooley-Tukey FFT
void computeFFT(float* real, float* imag, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
        int k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * (float)M_PI / (float)len;
        float wlen_r = cosf(angle);
        float wlen_i = sinf(angle);
        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float u_r = real[i + k];
                float u_i = imag[i + k];
                float v_r = real[i + k + len / 2] * w_r - imag[i + k + len / 2] * w_i;
                float v_i = real[i + k + len / 2] * w_i + imag[i + k + len / 2] * w_r;
                real[i + k] = u_r + v_r;
                imag[i + k] = u_i + v_i;
                real[i + k + len / 2] = u_r - v_r;
                imag[i + k + len / 2] = u_i - v_i;
                float next_w_r = w_r * wlen_r - w_i * wlen_i;
                w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
            }
        }
    }
}

// Streams TIME_FRAMES overlapped frames directly into model_input
void captureLibrosaSpectrogram() {
    float in_scale = model_input->params.scale;
    int in_zp = model_input->params.zero_point;

    // Prefill the first half of the sliding window
    memset(sliding_window, 0, sizeof(sliding_window));

    float max_power = 1e-12f;

    for (int t = 0; t < TIME_FRAMES; t++) {
        // Shift old samples back by HOP_LENGTH
        memmove(sliding_window, &sliding_window[HOP_LENGTH], (N_FFT - HOP_LENGTH) * sizeof(float));

        // Read next HOP_LENGTH samples (256) from I2S
        size_t bytes_read = 0;
        i2s_read(I2S_NUM_0, i2s_raw_buffer, sizeof(i2s_raw_buffer), &bytes_read, portMAX_DELAY);
        int read_count = bytes_read / sizeof(int32_t);

        for (int i = 0; i < read_count; i++) {
            sliding_window[(N_FFT - HOP_LENGTH) + i] = (float)(i2s_raw_buffer[i] >> 8) / 8388608.0f;
        }

        // ---------------------------------------------------------------
        // FIXED: periodic Hann window (divide by N_FFT, not N_FFT - 1).
        // librosa/scipy's STFT uses scipy.signal.get_window('hann', n_fft,
        // fftbins=True) by default, which is the periodic/DFT-even Hann
        // window. Dividing by (N_FFT - 1) is the *symmetric* Hann window
        // (e.g. numpy.hanning) and doesn't match what the model was
        // trained on.
        // ---------------------------------------------------------------
        for (int i = 0; i < N_FFT; i++) {
            float hanning = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)N_FFT));
            fft_real[i] = sliding_window[i] * hanning;
            fft_imag[i] = 0.0f;
        }

        computeFFT(fft_real, fft_imag, N_FFT);

        // Power spectrum
        for (int k = 0; k <= N_FFT / 2; k++) {
            power_spectrum[k] = (fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k]);
        }

        // Filterbank integration
        for (int m = 0; m < N_MELS; m++) {
            int left = mel_bin_indices[m];
            int center = mel_bin_indices[m + 1];
            int right = mel_bin_indices[m + 2];
            float mel_energy = 0.0f;

            for (int k = left; k < center; k++) {
                if (center != left) {
                    mel_energy += power_spectrum[k] * ((float)(k - left) / (float)(center - left));
                }
            }
            for (int k = center; k < right; k++) {
                if (right != center) {
                    mel_energy += power_spectrum[k] * ((float)(right - k) / (float)(right - center));
                }
            }

            mel_energy *= mel_norm[m]; // FIXED: apply Slaney-style filter normalization

            frame_mels[t][m] = mel_energy;
            if (mel_energy > max_power) {
                max_power = mel_energy;
            }
        }
    }

    // Power-to-dB, Min-Max scaling, and matching tensor format
    for (int t = 0; t < TIME_FRAMES; t++) {
        for (int m = 0; m < N_MELS; m++) {
            float power = frame_mels[t][m];

            // librosa.power_to_db(ref=np.max, top_db=80.0)
            float db = 10.0f * log10f(fmaxf(power, 1e-12f) / fmaxf(max_power, 1e-12f));
            if (db < -TOP_DB) db = -TOP_DB;

            // Normalize: (db + TOP_DB) / TOP_DB
            float norm_val = (db + TOP_DB) / TOP_DB;
            norm_val = fminf(fmaxf(norm_val, 0.0f), 1.0f);

            int quantized_val = (int)roundf(norm_val / in_scale) + in_zp;
            quantized_val = min(127, max(-128, quantized_val));

            // ---------------------------------------------------------------
            // FIXED: training data has shape (n_mels, time) -- librosa
            // returns mel-spectrograms that way, and the Colab notebook
            // never transposes it before feeding the CNN. The model's
            // input tensor axis 0 = mel, axis 1 = time. The previous
            // formula (t * N_MELS + m) wrote (time, mel) instead --
            // silently transposing every frame fed to the model. Since
            // N_MELS == TIME_FRAMES == 32 this never crashed or threw a
            // shape error, it just fed the CNN a rotated spectrogram.
            // ---------------------------------------------------------------
            model_input->data.int8[m * TIME_FRAMES + t] = (int8_t)quantized_val;
        }
    }
}

void setup() {
    // Query BEFORE Serial.begin() so nothing else has a chance to fragment
    // the heap first -- printed once Serial is up.
    size_t largest_block_at_start = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

    Serial.begin(115200);
    delay(1500);

    Serial.println("\n==================================================");
    Serial.println("   ESP32 Autoencoder Anomaly Detector (INT8)      ");
    Serial.println("==================================================");
    Serial.printf("Largest contiguous free block at very start of setup(): %d bytes\n", (int)largest_block_at_start);

    pinMode(LED_ALERT_PIN, OUTPUT);
    digitalWrite(LED_ALERT_PIN, LOW);

    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    if (tensor_arena == nullptr) {
        Serial.printf("\u274c ERROR: Failed to allocate %d KB tensor arena!\n", kTensorArenaSize / 1024);
        Serial.printf("Free heap: %d, Largest contiguous block now: %d\n",
                       (int)ESP.getFreeHeap(),
                       (int)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
        while (true) { delay(1000); }
    }
    Serial.printf("0. Allocated %d KB Tensor Arena from Internal Heap.\n", kTensorArenaSize / 1024);

    setupI2S();
    Serial.println("1. I2S Initialized.");

    const tflite::Model* model = tflite::GetModel(presigmoid_model_tflite);  // <-- rename to match your actual array name from the .h file
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("\u274c Schema mismatch! Model version: %d\n", model->version());
        while (true) { delay(1000); }
    }
    Serial.println("2. Model schema verified.");

    resolver.AddConv2D();
    // ---------------------------------------------------------------------
    // UPDATED: the decoder now uses UpSampling2D + Conv2D instead of
    // Conv2DTranspose, after a live on-device test showed TFLM computing a
    // different (wrong-direction) result than Python's TFLite interpreter
    // on the identical model and input -- a known class of INT8 correctness
    // issue with TransposeConv's embedded kernel. AddTransposeConv() is left
    // registered (harmless, unused) in case this firmware is still paired
    // with an older model export; AddResizeNearestNeighbor() is what the
    // new UpSampling2D-based model actually needs.
    // ---------------------------------------------------------------------
    resolver.AddTransposeConv();
    resolver.AddResizeNearestNeighbor();
    resolver.AddMaxPool2D();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddSub();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddLogistic();
    resolver.AddQuantize();
    resolver.AddDequantize();
    Serial.println("3. Ops registered.");

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &serial_reporter);
    interpreter = &static_interpreter;
    Serial.println("4. MicroInterpreter created.");

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("\u274c ERROR: AllocateTensors() failed!");
        while (true) { delay(1000); }
    }
    Serial.println("5. Tensors allocated.");

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    Serial.println("System Ready. Listening to microphone...\n");
}

// ---------------------------------------------------------------------------
// ADDED: diagnostic dump. Type 'd' + Enter in the Serial Monitor to capture
// the exact spectrogram tensor (dequantized, same values the model sees)
// on the NEXT completed cycle. Time it to fire while a fault condition is
// actively happening (start typing/sending right as you begin the physical
// fault action) or during steady normal operation, to compare directly
// against the notebook's spectrograms for the same audio type.
// ---------------------------------------------------------------------------
bool dump_requested = false;

void checkForDumpTrigger() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == 'd' || c == 'D') {
            dump_requested = true;
            Serial.println(">>> Dump armed -- will print on next completed cycle <<<");
        }
    }
}

void dumpSpectrogramIfRequested() {
    if (!dump_requested) return;
    dump_requested = false;

    // --- INPUT (the spectrogram fed into this exact inference) ---
    float in_scale2 = model_input->params.scale;
    int in_zp2 = model_input->params.zero_point;
    Serial.println("SPECTROGRAM_DUMP_START");
    Serial.printf("shape: %d mel x %d time\n", N_MELS, TIME_FRAMES);
    for (int m = 0; m < N_MELS; m++) {
        for (int t = 0; t < TIME_FRAMES; t++) {
            float val = (model_input->data.int8[m * TIME_FRAMES + t] - in_zp2) * in_scale2;
            Serial.print(val, 5);
            if (t < TIME_FRAMES - 1) Serial.print(",");
        }
        Serial.println();
    }
    Serial.println("SPECTROGRAM_DUMP_END");

    // ---------------------------------------------------------------------
    // ADDED: the model's OUTPUT (reconstruction) for this SAME inference --
    // needed to check whether TFLM's forward pass itself produces the same
    // result Python's interpreter would produce for identical input. Only
    // valid if this is called AFTER Invoke() has run for this cycle (see
    // loop() below -- the call site moved for exactly this reason).
    // ---------------------------------------------------------------------
    float out_scale2 = model_output->params.scale;
    int out_zp2 = model_output->params.zero_point;
    Serial.println("RECONSTRUCTION_DUMP_START");
    Serial.printf("shape: %d mel x %d time\n", N_MELS, TIME_FRAMES);
    for (int m = 0; m < N_MELS; m++) {
        for (int t = 0; t < TIME_FRAMES; t++) {
            // model_output is PRE-sigmoid (see loop() below) -- apply sigmoid
            // manually so this prints the actual reconstruction, not raw logits.
            float presigmoid_val2 = (model_output->data.int8[m * TIME_FRAMES + t] - out_zp2) * out_scale2;
            float val = 1.0f / (1.0f + expf(-presigmoid_val2));
            Serial.print(val, 5);
            if (t < TIME_FRAMES - 1) Serial.print(",");
        }
        Serial.println();
    }
    Serial.println("RECONSTRUCTION_DUMP_END");
}

void loop() {
    checkForDumpTrigger();
    captureLibrosaSpectrogram();

    unsigned long start_time = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long inference_time = millis() - start_time;

    if (invoke_status != kTfLiteOk) {
        Serial.println("\u274c Inference Invoke failed!");
        delay(200);
        return;
    }

    // MOVED: dump now happens AFTER Invoke(), so model_output reflects
    // this exact cycle's inference, not the previous one's stale result.
    dumpSpectrogramIfRequested();

    float in_scale = model_input->params.scale;
    int in_zp = model_input->params.zero_point;
    float out_scale = model_output->params.scale;
    int out_zp = model_output->params.zero_point;

    float reconstruction_mse = 0.0f;
    int total_elements = model_input->bytes;

    for (int i = 0; i < total_elements; i++) {
        float in_val = (model_input->data.int8[i] - in_zp) * in_scale;
        // model_output is now PRE-sigmoid (raw logits) -- the model deployed
        // here must be presigmoid_model, NOT the original fused-sigmoid model.
        // Applying sigmoid manually here bypasses the on-device Logistic
        // kernel bug confirmed via the checkpoint investigation.
        float presigmoid_val = (model_output->data.int8[i] - out_zp) * out_scale;
        float out_val = 1.0f / (1.0f + expf(-presigmoid_val));
        float diff = in_val - out_val;
        reconstruction_mse += diff * diff;
    }
    reconstruction_mse /= (float)total_elements;

    cycle_count++;
    if (cycle_count <= WARMUP_CYCLES) {
        Serial.printf("Warming up (%d/%d)... Raw: %.5f | Latency: %lu ms\n", cycle_count, WARMUP_CYCLES, reconstruction_mse, inference_time);
        return; // don't push into history, don't evaluate threshold
    }

    float smoothed_mse = pushAndGetSmoothedMSE(reconstruction_mse);

    // ---------------------------------------------------------------------
    // UPDATED: two-sided check. HIGH-side triggers correspond to faults that
    // add complexity to the sound (bearing rub, blade strike, impacts) --
    // LOW-side triggers correspond to faults that make the sound simpler /
    // more tonal than normal (airflow obstruction, speed instability). The
    // direction is logged since it's a genuine hint at fault category, not
    // just a debugging detail.
    // ---------------------------------------------------------------------
    bool is_low_anomaly = smoothed_mse < ANOMALY_THRESHOLD_LOW;
    bool is_high_anomaly = smoothed_mse > ANOMALY_THRESHOLD_HIGH;

    if (is_low_anomaly || is_high_anomaly) {
        anomaly_streak++;
    } else {
        anomaly_streak = max(0, anomaly_streak - 1);
    }

    if (anomaly_streak >= CONSECUTIVE_ANOMALIES_REQUIRED) {
        digitalWrite(LED_ALERT_PIN, HIGH);
        const char* direction = is_low_anomaly ? "LOW (possible obstruction/speed fault)" : "HIGH (possible bearing/impact fault)";
        Serial.printf("\u26a0\ufe0f  FAULT DETECTED [%s] | Raw: %.5f | Smoothed: %.5f | Latency: %lu ms\n", direction, reconstruction_mse, smoothed_mse, inference_time);
    } else {
        digitalWrite(LED_ALERT_PIN, LOW);
        Serial.printf("Status: NORMAL | Raw: %.5f | Smoothed: %.5f | Latency: %lu ms\n", reconstruction_mse, smoothed_mse, inference_time);
    }
}
