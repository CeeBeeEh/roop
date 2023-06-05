import cv2
from roop.codeformer import process_face_enhance


def get_face_from_frame(bbox, frame):
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]

    # bbox[0] = bbox[0] - 40
    # bbox[1] = bbox[1] - 40
    # bbox[2] = bbox[2] + 40
    # bbox[3] = bbox[3] + 40
    # if bbox[1] < 0:
    #     bbox[1] = 0
    # if bbox[0] < 0:
    #     bbox[0] = 0
    # if bbox[2] > frame.shape[0]:
    #     bbox[2] = frame.shape[0]
    # if bbox[3] > frame.shape[1]:
    #     bbox[3] = frame.shape[1]

    bbox[0] = max(bbox[0] - (h / 2), 0)
    bbox[1] = max(bbox[1] - (w / 4), 0)
    bbox[2] = min(bbox[2] + (h / 2), frame.shape[1])
    bbox[3] = min(bbox[3] + (w / 4), frame.shape[0])

    return bbox, frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]


def enhance_face(bbox, frame):
    bbox, face = get_face_from_frame(bbox, frame)
    try:
        cv2.imwrite('/mnt/4TB/Workspace/DeepFaceLab/DumbAndDumber/face_test_pre.jpg', face)
    except Exception as error:
        print(error)

    face = process_face_enhance(face)
    w = int(bbox[3] - bbox[1])
    h = int(bbox[2] - bbox[0])
    print(h, w)
    face = cv2.resize(face, (h, w), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite('/mnt/4TB/Workspace/DeepFaceLab/DumbAndDumber/face_test.jpg', face)

    # blurred_img = cv2.GaussianBlur(face, (21, 21), 0)
    # mask = np.zeros(face.shape, np.uint8)
    #
    # gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[2]
    # contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # cv2.drawContours(mask, contours, -1, (255, 255, 255), 5)
    # output = np.where(mask == np.array([255, 255, 255]), blurred_img, face)

    frame[int(bbox[1]):int(bbox[1]) + face.shape[0], int(bbox[0]):int(bbox[0]) + face.shape[1]] = face
    # frame = cv2.addWeighted(frame, 1, face, 1, 0)
    return frame
