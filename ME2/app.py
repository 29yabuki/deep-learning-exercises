#!/usr/bin/env python3
import gradio as gr
import os, time, torch, numpy as np
from ultralytics import YOLO

loaded_model_live = None
current_model_name_live = None
current_imgsz_live = None
selected_device = None
is_half_supported = False
last_process_time = 0.0
last_processed_frame_live = None
last_timing_str_live = "Initializing..."
THROTTLE_INTERVAL = 0.06

model_path_prefix = "models/"
available_models = [
    "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt",
    "yolo11l-pose.pt", "yolo11x-pose.pt"
]
DEFAULT_MODEL  = "yolo11n-pose.pt"
DEFAULT_IMGSZ  = 320

def setup_device():
    global selected_device, is_half_supported
    if torch.cuda.is_available():
        selected_device = 0
        is_half_supported = torch.cuda.get_device_capability(0) >= (7, 0)
    else:
        selected_device, is_half_supported = "cpu", False
    return selected_device

def load_or_get_model_live(model_name, imgsz):
    global loaded_model_live, current_model_name_live, current_imgsz_live
    global last_processed_frame_live, last_timing_str_live

    if selected_device is None:
        setup_device()

    if (loaded_model_live is None or current_model_name_live != model_name
            or current_imgsz_live != imgsz):
        last_processed_frame_live, last_timing_str_live = None, f"Loading {model_name}…"
        try:
            if loaded_model_live is not None:
                del loaded_model_live
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            loaded_model_live = YOLO(os.path.join(model_path_prefix, model_name))
            current_model_name_live, current_imgsz_live = model_name, imgsz
            dummy = np.zeros((imgsz, imgsz, 3), np.uint8)
            loaded_model_live.predict(source=dummy, device=selected_device,
                                      imgsz=imgsz, half=is_half_supported, verbose=False, stream=True)
            last_timing_str_live = "Ready"
        except Exception as e:
            last_timing_str_live = "Model Load Error"
            raise gr.Error(f"Failed to load {model_name}: {e}")
    return loaded_model_live

def process_live_frame(frame, conf, iou, model_name, imgsz, show_time, pause):
    global last_process_time, last_processed_frame_live, last_timing_str_live
    if pause:
        return frame, "Processing Paused"

    if time.perf_counter() - last_process_time < THROTTLE_INTERVAL:
        return (last_processed_frame_live or frame,
                last_timing_str_live + " (Throttled)")

    last_process_time = time.perf_counter()
    if frame is None:
        last_timing_str_live = "No Frame Input"
        return last_processed_frame_live, last_timing_str_live

    try:
        model = load_or_get_model_live(model_name, imgsz)
        start = time.perf_counter()
        res = next(model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz,
                                 half=is_half_supported, device=selected_device,
                                 verbose=False, save=False, show_labels=True,
                                 show_conf=True, stream=True))
        infer_ms = (time.perf_counter() - start) * 1000
        last_processed_frame_live = res.plot()
        last_timing_str_live = f"Infer: {infer_ms:.0f} ms" if show_time else ""
        return last_processed_frame_live, last_timing_str_live
    except Exception:
        last_timing_str_live = "Prediction Error!"
        return last_processed_frame_live or frame, last_timing_str_live

def ping_server():
    return True

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# ME 2: YOLO‑11 Inference UI")
    gr.Markdown("By Sean Caranzo")

    with gr.Row():
        live_input  = gr.Image(sources=["webcam"], type="numpy",
                               streaming=True, label="Webcam")
        live_output = gr.Image(type="numpy", interactive=False, label="Output")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Configuration")
            conf   = gr.Slider(0, 1, 0.40, 0.05, label="Confidence Threshold")
            iou    = gr.Slider(0, 1, 0.50, 0.05, label="IoU Threshold")
            model  = gr.Radio(available_models, value=DEFAULT_MODEL,
                              label="Select YOLO Model")
            imgsz  = gr.Slider(320, 640, DEFAULT_IMGSZ, 32,
                               label="Live Image Size")
            pause  = gr.Checkbox(False, label="Pause Live")
            timing = gr.Checkbox(True,  label="Show Timing Info")
            btn    = gr.Button("Check Latency")
            rtt    = gr.Textbox(label="RTT (ms)", interactive=False)

    live_input.stream(
        fn=process_live_frame,
        inputs=[live_input, conf, iou, model, imgsz, timing, pause],
        outputs=[live_output, rtt]
    )

    dummy_state = gr.State()
    btn.click(fn=ping_server, inputs=[], outputs=[dummy_state],
              js="() => { window.t0 = performance.now(); }"
    ).then(fn=None, inputs=None, outputs=[rtt],
           js="() => (performance.now() - window.t0).toFixed(1) + ' ms'")

if __name__ == "__main__":
    setup_device()
    load_or_get_model_live(DEFAULT_MODEL, DEFAULT_IMGSZ)
    demo.queue().launch(share=True, debug=True)