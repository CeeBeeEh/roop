import onnxruntime

all_faces = None
log_level = 'error'
cpu_cores = None
gpu_threads = None
gpu_vendor = None
providers = onnxruntime.get_available_providers()
face_enhance_scale = 2
face_enhance_detection_model = 'retinaface_resnet50'
face_enhance_bg_tile = 400
face_enhance_fidelity_weight = 0.7

if 'TensorrtExecutionProvider' in providers:
    providers.remove('TensorrtExecutionProvider')
