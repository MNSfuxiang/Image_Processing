import numpy as np
import cv2
import math


def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf

def Kernel_generate(img_size, move_angle):
    PSF = np.zeros(shape=img_size)
    center_position = (img_size[0]+1)/2
    slope_tan = math.tan(move_angle*math.pi/180)
    slope_cot = 1/slope_tan
    if slope_tan<=1:
        for i in range(15):
            offset = round(i*slope_tan)
            PSF[int(center_position+offset), int(center_position-offset)] = 1
        return PSF/PSF.sum()
    else:
        for i in range(15):
            offset = round(i*slope_cot)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF/PSF.sum()


def IMG_blurred(in_img, PSF, eps):
    in_fft = np.fft.fft2(in_img)
    PSF_fft = np.fft.fft2(PSF)+eps
    blurred = np.fft.ifft2(in_fft*PSF_fft)
    blurred = np.abs(np.fft.fftshift(blurred))
    # cv2.imwrite('./image_data/blured.jpg', np.asarray(blurred).astype(int))
    return blurred


def wiener(input_img, PSF, eps, K = 0.001):
    in_fft = np.fft.fft2(input_img)
    PSF_fft_1 = np.fft.fft2(PSF)+eps
    PSF_fft_2 = np.conj(PSF_fft_1)/(np.abs(PSF_fft_1)**2 + K)
    recover_img = np.fft.ifft2(in_fft*PSF_fft_2)
    recover_img = np.abs(np.fft.fftshift(recover_img))
    # cv2.imwrite('./image_data/wiener.jpg', np.asarray(recover_img).astype(int))
    return recover_img

def CLSF(blurred, PSF, eps, gamma = 0.001):
    out_h, out_w = blurred.shape[:2]
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    PF_kernel = psf2otf(kernel,[out_h, out_w])
    in_noise = np.fft.fft2(blurred)
    numerator = np.conj(np.fft.fft2(PSF)+eps)
    denominator = (np.fft.fft2(PSF)+eps) ** 2 + gamma * (PF_kernel ** 2)
    recover_img = np.fft.ifft2(numerator * in_noise / denominator)
    recover_img = np.abs(np.fft.fftshift(recover_img))
    return recover_img



def inverse(input_img, PSF, eps):
    in_fft = np.fft.fft2(input_img)
    PSF_fft = np.fft.fft2(PSF) + eps
    recover_img = np.fft.ifft2(in_fft/PSF_fft)
    recover_img = np.abs(np.fft.fftshift(recover_img))
    # cv2.imwrite('./image_data/inverse.jpg', np.asarray(recover_img).astype(int))
    return recover_img


def inverse_limite_R(input_img, PSF, eps, R):
    out_h, out_w = input_img.shape[:2]
    in_fft = np.fft.fftshift(np.fft.fft2(input_img))
    PSF_fft = np.fft.fft2(PSF) + eps
    H = np.zeros(shape=(out_h, out_w))
    for x in range(-int(out_h/2), int(out_h/2)):
        for y in range(-int(out_w/2), int(out_w/2)):
            R_n = (x**2+y**2)**0.5
            H[int(x+(out_h/2)+1),int(y+(out_w/2)+1)] = 1/(1+(R_n/R)**20)
    recover_img = np.fft.ifft2(in_fft*H / PSF_fft)
    recover_img = np.abs(np.fft.ifftshift(recover_img))
    return recover_img



if __name__ == '__main__':
    in_img = cv2.imread('./img_data/source.jpg')
    b_gray, g_gray, r_gray = cv2.split(in_img.copy())
    blurred_img = np.zeros(in_img.shape)
    wiener_img = np.zeros(in_img.shape)
    inverse_img = np.zeros(in_img.shape)
    inverse_limite_R_img = np.zeros(in_img.shape)
    CLSF_img = np.zeros(in_img.shape)
    channel = 0
    for gray in [b_gray, g_gray, r_gray]:
        img_h, img_w = gray.shape[:2]
        PSF = Kernel_generate((img_h, img_w), 30)
        blurred_img[:, :, channel] = IMG_blurred(gray, PSF, 0.001)
        inverse_img[:, :, channel] = inverse(blurred_img[:, :, channel], PSF, 0.001)
        inverse_limite_R_img[:, :, channel] = inverse_limite_R(blurred_img[:, :, channel], PSF, 0.001, 60)
        CLSF_img[:, :, channel] = CLSF(blurred_img[:, :, channel], PSF, 0.001)
        wiener_img[:, :, channel] = wiener(blurred_img[:, :, channel], PSF, 0.001)
        channel += 1
    cv2.imwrite('./img_data/blured.jpg',
                cv2.merge([blurred_img[:, :, 0], blurred_img[:, :, 1], blurred_img[:, :, 2]]))
    cv2.imwrite('./img_data/wiener.jpg',
                cv2.merge([wiener_img[:, :, 0], wiener_img[:, :, 1], wiener_img[:, :, 2]]))
    cv2.imwrite('./img_data/inverse.jpg',
                cv2.merge([inverse_img[:, :, 0], inverse_img[:, :, 1], inverse_img[:, :, 2]]))
    cv2.imwrite('./img_data/inverse_limite_.jpg', cv2.merge(
        [inverse_limite_R_img[:, :, 0], inverse_limite_R_img[:, :, 1], inverse_limite_R_img[:, :, 2]]))
    cv2.imwrite('./img_data/CLSF.jpg', cv2.merge([CLSF_img[:, :, 0], CLSF_img[:, :, 1], CLSF_img[:, :, 2]]))
