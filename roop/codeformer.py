import torch
import roop.globals
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

CF_NET = None
FACE_HELPER = None
FACE_ENHANCER = None
DEVICE = None


def get_cf_device():
    global DEVICE
    if DEVICE is None:
        if True:
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            DEVICE = device = torch.device('cpu')
    return DEVICE


def get_codeformer():
    global CF_NET
    if CF_NET is None:
        CF_NET = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                                 connect_list=['32', '64', '128', '256']).to(get_cf_device())
        ckpt_path = 'codeformer.pth'
        checkpoint = torch.load(ckpt_path)['params_ema']
        CF_NET.load_state_dict(checkpoint)
        CF_NET.eval()
    return CF_NET


def get_face_helper():
    global FACE_HELPER
    if FACE_HELPER is None:
        FACE_HELPER = FaceRestoreHelper(
            roop.globals.face_enhance_scale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=roop.globals.face_enhance_detection_model,
            save_ext='png',
            use_parse=True,
            device=get_cf_device())
    return FACE_HELPER


def get_realesrgan():
    global FACE_ENHANCER
    if FACE_ENHANCER is None:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer

        use_half = False
#        if roop.globals.use_gpu:
#            if torch.cuda.is_available():  # set False in CPU/MPS mode
#                no_half_gpu_list = ['1650', '1660']  # set False for GPUs that don't support f16
#                if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
#                    use_half = True

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        model.to(get_cf_device())
        print('FACE_ENHANCER = RealESRGANer')
        FACE_ENHANCER = RealESRGANer(
            scale=2,
            model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
            model=model,
            tile=roop.globals.face_enhance_bg_tile,
            tile_pad=40,
            pre_pad=0,
            half=use_half
        )

        if not True == False:  # CPU
            import warnings
            warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                          'The unoptimized RealESRGAN is slow on CPU. '
                          'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                          category=RuntimeWarning)
    return FACE_ENHANCER


def process_face_enhance(face):
    try:
        torch.cuda.set_per_process_memory_fraction(0.4, 0)
        get_face_helper().clean_all()
        get_face_helper().read_image(face)
        num_det_faces = get_face_helper().get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5)
        get_face_helper().align_warp_face()
        cropped_face = get_face_helper().cropped_faces.pop()

        face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        face_t = face_t.unsqueeze(0).to(get_cf_device())
        with torch.no_grad():
            output = get_codeformer()(face_t, w=0.95, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        del output
        torch.cuda.empty_cache()
        restored_face = restored_face.astype('uint8')
        get_face_helper().add_restored_face(restored_face, cropped_face)

        bg_img = get_realesrgan().enhance(face, outscale=2)[0]
        get_face_helper().get_inverse_affine(None)
        print('restored_img = get_face_helper()')
        restored_img = get_face_helper().paste_faces_to_input_image(upsample_img=bg_img, draw_box=False,
                                                                    face_upsampler=get_realesrgan())

        return restored_img
    except Exception as error:
        print(f'\tFailed inference for CodeFormer: {error}')

    return face
