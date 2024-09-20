# coding: utf-8
import numpy as np
from scipy.ndimage.interpolation import rotate
import torch
import util


def random_noise(image):
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    if image.ndim == 4:
        n, c, h, w = image.shape
        noise = torch.normal(mean=mean, std=sigma, size=(n, c, h, w))
    elif image.ndim == 2:
        n, w = image.shape
        noise = torch.normal(mean=mean, std=sigma, size=(n, w))

    image = image + util.to_device(noise.float())

    return image


def horizontal_flip(image):
    n, _, _, _ = image.shape
    image2 = image.clone()

    rand = np.random.rand(n)
    reverse = torch.arange(image.shape[3] - 1, -1, -1)

    for i in range(n):
        if rand[i] < 0.5:
            image2[i] = image[i, :, :, reverse]

    return image2


def vertical_flip(image):
    n, _, _, _ = image.shape
    image2 = image.clone()

    rand = np.random.rand(n)
    reverse = torch.arange(image.shape[2] - 1, -1, -1)

    for i in range(n):
        if rand[i] < 0.5:
            image2[i] = image[i, :, reverse, :]

    return image2


def random_crop(image):
    n, c, h, w = image.shape
    image2 = torch.zeros(n, c, h, w).cuda()

    crop_size = (4 * h // 5, 4 * w // 5)

    for i in range(n):
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])

        bottom = top + crop_size[0]
        right = left + crop_size[1]

        x = image[i, :, top:bottom, left:right].clone()
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])

        image2[i] = torch.nn.functional.upsample(x, size=(h, w), mode="bilinear", align_corners=True)

    return image2


def random_translation(image):
    r1 = 0.2
    r2 = 0.2

    n, c, h, w = image.shape
    image2 = torch.zeros(n, c, h, w).cuda()

    offset = np.random.randint(-h//5, h//5 + 1, size=(n, 2))
    # offset1 = int(h * r1)
    # offset2 = int(h * r2)

    for i in range(n):
        offset1, offset2 = offset[i]

        left = max(0, offset1)
        top = max(0, offset2)
        right = min(w, w + offset1)
        bottom = min(h, h + offset2)

        image2[i, :, top - offset2:bottom - offset2, left - offset1:right - offset1] = image[i, :, top:bottom, left:right]

    return image2


def mixup(image, label, num_classes, alpha=0):
    if alpha == 0:
        alpha = 1.0

    rand_idx = torch.randperm(label.shape[0])
    image2 = image[rand_idx].clone()
    label2 = label[rand_idx].clone()

    y_one_hot = torch.eye(num_classes, device='cuda')[label]
    y2_one_hot = torch.eye(num_classes, device='cuda')[label2]
    mix_rate = np.random.beta(alpha, alpha, image.shape[0])

    mix_rate2 = None
    if image.ndim == 2:
        mix_rate2 = util.to_device(torch.from_numpy(mix_rate.reshape((image.shape[0], 1))).float())
    elif image.ndim == 4:
        mix_rate2 = util.to_device(torch.from_numpy(mix_rate.reshape((image.shape[0], 1, 1, 1))).float())

    mix_rate = util.to_device(torch.from_numpy(mix_rate.reshape((image.shape[0], 1))).float())

    x_mixed = image.clone() * mix_rate2 + image2.clone() * (1 - mix_rate2)
    y_soft = y_one_hot * mix_rate + y2_one_hot * (1 - mix_rate)

    return x_mixed, y_soft


def cutout(image, scale=0):
    if scale == 0:
        scale = 2.0

    image2 = image.clone()
    mask_value = 0

    if image.ndim == 4:
        n, _, h, w = image.shape
        mask_size = int(h // scale)

        for i in range(n):
            top = np.random.randint(0 - mask_size // 2, h - mask_size // 2)
            left = np.random.randint(0 - mask_size // 2, w - mask_size // 2)
            bottom = top + mask_size
            right = left + mask_size

            if top < 0:
                top = 0
            if left < 0:
                left = 0
            if bottom > h:
                bottom = h
            if right > w:
                right = w

            image2[i][:, top:bottom, left:right] = mask_value

    elif image.ndim == 2:
        n, w = image.shape
        mask_size = int(w // scale)
        for i in range(n):
            left = np.random.randint(0 - mask_size // 2, w - mask_size // 2)
            right = left + mask_size

            if left < 0:
                left = 0
            if right > w:
                right = w
            image2[i][left:right] = mask_value

    return image2


def random_erasing(image, p=0.5, s=(0.02, 0.4), r=(0.3, 3)):
    if np.random.rand() > p:
       return image

    image2 = image.clone()

    n, _, h, w = image2.shape

    for i in range(n):
        mask_value = np.random.rand()

        mask_area = np.random.randint(h * w * s[0], h * w * s[1])

        mask_aspect_ratio = np.random.rand() * r[1] + r[0]

        mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
        if mask_height > h - 1:
            mask_height = h - 1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w - 1:
            mask_width = w - 1

        top = np.random.randint(0, h - mask_height)
        left = np.random.randint(0, w - mask_width)
        bottom = top + mask_height
        right = left + mask_width

        image2[i][:, top:bottom, left:right] = mask_value

    return image


def cutmix(image, label, num_classes, alpha=0):
    if alpha == 0:
        alpha = 0.5

    rand_idx = torch.randperm(label.shape[0])
    image2 = image[rand_idx].clone()
    label2 = label[rand_idx].clone()

    y_one_hot = torch.eye(num_classes, device='cuda')[label]
    y2_one_hot = torch.eye(num_classes, device='cuda')[label2]

    mix_rate = np.random.beta(alpha, alpha, image.shape[0])

    if image2.ndim == 4:
        n, _, h, w = image2.shape

        for i in range(n):
            r_x = np.random.randint(w)
            r_y = np.random.randint(h)
            r_l = np.sqrt(mix_rate[i])
            r_w = np.int(w * r_l)
            r_h = np.int(h * r_l)
            bx1 = np.int(np.clip(r_x - r_w, 0, w))
            by1 = np.int(np.clip(r_y - r_h, 0, h))
            bx2 = np.int(np.clip(r_x + r_w, 0, w))
            by2 = np.int(np.clip(r_y + r_h, 0, h))

            image2[i][:, bx1:bx2, by1:by2] = image[i][:, bx1:bx2, by1:by2]

        mix_rate = util.to_device(torch.from_numpy(mix_rate.reshape((image.shape[0], 1))).float())
        new_label = mix_rate * y_one_hot + (1 - mix_rate) * y2_one_hot

    return image2, new_label


def random_rotation(image):
    angle = 30

    size, c, h, w = image.shape
    image2 = image.clone()
    image2 = np.array(image2.data.cpu())

    if c == 1:
        image2 = np.squeeze(image2)
    else:
        image2 = image2.transpose((0, 2, 3, 1))

    for i in range(size):
        # angle = np.random.randint(180)
        image_rotate = rotate(image2[i], angle)

        if c == 1:
            image2[i] = imresize(image_rotate, (h, w))
        else:
            image2[i] = imresize(image_rotate, (h, w, c))
    if c == 1:
        image2 = image2[:, np.newaxis, :, :] / 255.0
    else:
        image2 = image2.transpose((0, 3, 1, 2)) / 255.0

    image2 = torch.from_numpy(image2).float()

    return image2


def ch_contrast(image):
    a = 0.5

    if image.ndim == 2:
        rate = torch.rand(image.shape[0], 1) + a
    else:
        rate = torch.rand(image.shape[0], image.shape[1], 1, 1) + a

    return image * util.to_device(rate.float())