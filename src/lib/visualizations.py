"""
Visualization functions
"""

import itertools
from math import ceil
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import colors
import imageio
from webcolors import name_to_rgb
from torchvision.utils import draw_segmentation_masks

from CONFIG import COLORS


def visualize_sequence(sequence, savepath=None,  tag="sequence", add_title=True,
                       add_axis=False, n_cols=10, font_size=11, n_channels=3,
                       titles=None, tb_writer=None, iter=0, **kwargs):
    """ Visualizing a grid with several images/frames """

    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols)

    figsize = kwargs.pop("figsize", (3*n_cols, 3*n_rows))
    fig.set_size_inches(*figsize)
    if("suptitle" in kwargs):
        fig.suptitle(kwargs["suptitle"])
        del kwargs["suptitle"]

    ims = []
    fs = []
    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_rows > 1 else ax[col]
        f = sequence[i].permute(1, 2, 0).cpu().detach().clamp(0, 1)
        if(n_channels == 1):
            f = f[..., 0]
        im = a.imshow(f, **kwargs)
        ims.append(im)
        fs.append(f)
        if(add_title):
            if(titles is not None):
                cur_title = "" if i >= len(titles) else titles[i]
                a.set_title(cur_title, fontsize=font_size)
            else:
                a.set_title(f"Frame {i}", fontsize=font_size)

    # removing axis
    if(not add_axis):
        for row in range(n_rows):
            for col in range(n_cols):
                a = ax[row, col] if n_rows > 1 else ax[col]
                if n_cols * row + col >= n_frames:
                    a.axis("off")
                else:
                    a.set_yticks([])
                    a.set_xticks([])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if tb_writer is not None:
        img_grid = torch.stack(fs).permute(0, 3, 1, 2)
        tb_writer.add_images(fig_name=tag, img_grid=img_grid, step=iter)
    return fig, ax, ims



def visualize_recons(imgs, recons, savepath=None,  tag="recons", n_cols=10,
                     tb_writer=None, iter=0):
    """ Visualizing original imgs, recons and error """
    B, C, H, W = imgs.shape
    imgs = imgs.cpu().detach()
    recons = recons.cpu().detach()
    n_cols = min(B, n_cols)

    fig, ax = plt.subplots(nrows=3, ncols=n_cols)
    fig.set_size_inches(w=n_cols * 3, h=3 * 3)
    for i in range(n_cols):
        a = ax[:, i] if n_cols > 1 else ax
        a[0].imshow(imgs[i].permute(1, 2, 0).clamp(0, 1))
        a[1].imshow(recons[i].permute(1, 2, 0).clamp(0, 1))
        err = (imgs[i] - recons[i]).sum(dim=-3)
        a[2].imshow(err, cmap="coolwarm", vmin=-1, vmax=1)
        a[0].axis("off")
        a[1].axis("off")
        a[2].axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if tb_writer is not None:
        tb_writer.add_images(
                fig_name=f"{tag}_imgs", img_grid=np.array(imgs), step=iter
            )
        tb_writer.add_images(
                fig_name=f"{tag}_recons", img_grid=np.array(recons), step=iter
            )

    plt.close(fig)
    return



def visualize_decomp(objs, savepath=None, tag="decomp", vmin=0, vmax=1,
                     add_axis=False, size=3, n_cols=10, tb_writer=None,
                     iter=0, ax_labels=None, **kwargs):
    """
    Visualizing object/mask decompositions, having one obj-per-row

    Args:
    -----
    objs: torch Tensor
        decoded decomposed objects or masks. Shape is (B, Num Objs, C, H, W)
    """
    B, N, C, H, W = objs.shape
    n_channels = C
    if B > n_cols:
        objs = objs[:n_cols]
    else:
        n_cols = B
    objs = objs.cpu().detach()

    ims = []
    fs = []
    fig, ax = plt.subplots(nrows=N, ncols=n_cols)
    fig.set_size_inches(w=n_cols * size, h=N * size)
    for col in range(n_cols):
        for row in range(N):
            if N == 1 and n_cols > 1:
                a = ax[col]
            elif N > 1 and n_cols == 1:
                a = ax[row]
            else:
                a = ax[row, col]
            f = objs[col, row].permute(1, 2, 0).clamp(vmin, vmax)
            fim = f.clone()
            if(n_channels == 1):
                fim = fim.repeat(1, 1, 3)
            im = a.imshow(fim, **kwargs)
            ims.append(im)
            fs.append(f)

    for col in range(n_cols):
        if N == 1 and n_cols > 1:
            a = ax[col]
        elif n_cols == 1 and N > 1:
            a = ax[0]
        else:
            a = ax[0, col]
        a.set_title(f"#{col+1}")

    # removing axis
    if(not add_axis):
        for row in range(N):
            for col in range(n_cols):
                if N == 1 and n_cols > 1:
                    a = ax[col]
                elif N > 1 and n_cols == 1:
                    a = ax[row]
                else:
                    a = ax[row, col]
                cmap = kwargs.get("cmap", "")
                a.set_xticks([])
                a.set_yticks([])
            if ax_labels is not None:
                a = ax[row, 0] if n_cols > 1 else ax[row]
                a.set_ylabel(ax_labels[row])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if tb_writer is not None:
        tb_writer.add_figure(tag=tag, figure=fig, step=iter)
    return fig, ax, ims




def visualize_qualitative_eval(context, targets, preds, savepath=None,
                               context_titles=None, suptitle=None,
                               target_titles=None, pred_titles=None, fontsize=16):
    """
    Qualitative evaluation of one example. Simultaneuosly visualizing context, ground truth
    and predicted frames.
    """
    n_context = context.shape[0]
    n_targets = targets.shape[0]
    n_preds = preds.shape[0]

    n_cols = min(10, max(n_targets, n_context))
    n_rows = 1 + ceil(n_preds / n_cols) + ceil(n_targets / n_cols)
    n_rows_pred = 1 + ceil(n_targets / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(w=n_cols*4, h=(n_rows+1)*4)

    context = add_border(
            x=context, color_name="green", pad=2
        ).permute(0, 2, 3, 1).cpu().detach()
    targets = add_border(
            x=targets, color_name="green", pad=2
        ).permute(0, 2, 3, 1).cpu().detach()
    preds = add_border(
            x=preds, color_name="red", pad=2
        ).permute(0, 2, 3, 1).cpu().detach()

    if context_titles is None:
        ax[0, n_cols//2].set_title("Seed Frames", fontsize=fontsize)
    if target_titles is None:
        ax[1, n_cols//2].set_title("Target Frames", fontsize=fontsize)
    if pred_titles is None:
        ax[n_rows_pred, n_cols//2].set_title("Predicted Frames", fontsize=fontsize)
    
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)

    for i in range(n_context):
        ax[0, i].imshow(context[i].clamp(0, 1))
        if context_titles is not None:
            ax[0, i].set_title(context_titles[i])
    for i in range(n_preds):
        cur_row, cur_col = i // n_cols, i % n_cols
        if i < n_targets:
            ax[1 + cur_row, cur_col].imshow(targets[i].clamp(0, 1))
            if target_titles is not None:
                ax[1 + cur_row, cur_col].set_title(target_titles[i])
        if i < n_preds:
            ax[n_rows_pred + cur_row, cur_col].imshow(preds[i].clamp(0, 1))
            if pred_titles is not None:
                ax[n_rows_pred + cur_row, cur_col].set_title(pred_titles[i])

    for a_row in ax:
        for a_col in a_row:
            a_col.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return fig, ax



def add_border(x, color_name, pad=1):
    """
    Adding border to image frames

    Args:
    -----
    x: numpy array
        image to add the border to
    color_name: string
        Name of the color to use
    pad: integer
        number of pixels to pad each side
    """
    nc, h, w = x.shape[-3:]
    b = x.shape[:-3]

    zeros = torch.zeros if torch.is_tensor(x) else np.zeros
    px = zeros((*b, 3, h+2*pad, w+2*pad))
    color = colors.to_rgb(color_name)
    px[..., 0, :, :] = color[0]
    px[..., 1, :, :] = color[1]
    px[..., 2, :, :] = color[2]
    if nc == 1:
        for c in range(3):
            px[..., c, pad:h+pad, pad:w+pad] = x[:, 0]
    else:
        px[..., pad:h+pad, pad:w+pad] = x
    return px



def visualize_aligned_slots(recons_objs, savepath=None, fontsize=16, mult=3):
    """
    Visualizing the reconstructed objects after alignment of slots.

    Args:
    -----
    recons_objs: torch Tensor
        Reconstructed objects (objs * masks) for a sequence after alignment.
        Shape is (num_frames, num_objs, C, H, W)
    """
    T, N, _, _, _ = recons_objs.shape

    fig, ax = plt.subplots(nrows=N, ncols=T)
    fig.set_size_inches((T * mult, N * mult))
    for t_step in range(T):
        for slot_id in range(N):
            ax[slot_id, t_step].imshow(
                    recons_objs[t_step, slot_id].cpu().detach().clamp(0, 1).permute(1, 2, 0),
                    vmin=0,
                    vmax=1
                )
            if t_step == 0:
                ax[slot_id, t_step].set_ylabel(f"Object {slot_id + 1}", fontsize=fontsize)
            if slot_id == N-1:
                ax[slot_id, t_step].set_xlabel(f"Time Step {t_step + 1}", fontsize=fontsize)
            if slot_id == 0:
                ax[slot_id, t_step].set_title(f"Time Step {t_step + 1}", fontsize=fontsize)
            ax[slot_id, t_step].set_xticks([])
            ax[slot_id, t_step].set_yticks([])
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return fig, ax



def make_gif(frames, savepath, n_seed=4, use_border=False):
    """ Making a GIF with the frames """
    with imageio.get_writer(savepath, mode='I', loop=0) as writer:
        for i, frame in enumerate(frames):
            frame = torch.nn.functional.interpolate(
                    frame.unsqueeze(0),
                    scale_factor=2
                )[0]  # HACK to have higher resolution GIFs
            up_frame = frame.cpu().detach().clamp(0, 1)
            if use_border:
                color_name = "green" if i < n_seed else "red"
                disp_frame = add_border(up_frame, color_name=color_name, pad=2)
            else:
                disp_frame = up_frame
            disp_frame = (disp_frame * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            writer.append_data(disp_frame)



def visualize_metric(vals, start_x=0, title=None, xlabel=None,
                     savepath=None, **kwargs):
    """ Function for visualizing the average metric per frame """
    fig, ax = plt.subplots(1, 1)
    ax.plot(vals, linewidth=3)
    ax.set_xticks(
            ticks=np.arange(len(vals)),
            labels=np.arange(start=start_x, stop=len(vals) + start_x)
        )
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    plt.close(fig)
    return



def idx_to_one_hot(x):
    """
    Converting from instance indices into instance-wise one-hot encodings
    """
    num_classes = x.unique().max() + 1
    shape = x.shape
    x = x.flatten().to(torch.int64).view(-1,)
    y = torch.nn.functional.one_hot(x, num_classes=num_classes)
    y = y.view(*shape, num_classes)  # (..., Height, Width, Classes)
    y = y.transpose(-3, -1).transpose(-2, -1)  # (..., Classes, Height, Width)
    return y



def masks_to_rgb(x):
    """ Converting masks to RGB images for visualization """
    # we make the assumption that background is the mask with the most pixels
    num_objs = x.unique().max()
    background_val = x.flatten(-2).mode(dim=-1)[0]
    colors = COLORS

    imgs = []
    for i in range(x.shape[0]):
        img = torch.zeros(*x.shape[1:], 3)
        for cls in range(num_objs + 1):
            color = colors[cls+1] if cls != background_val[i] else "seashell"
            color_rgb = torch.tensor(name_to_rgb(color)).float()
            img[x[i] == cls, :] = color_rgb / 255
        imgs.append(img)
    imgs = torch.stack(imgs)
    imgs = imgs.transpose(-3, -1).transpose(-2, -1)
    return imgs



def overlay_segmentations(frames, segmentations, colors, num_classes=None, alpha=0.7):
    """
    Overlaying the segmentation on a sequence of images
    """
    if num_classes is None:
        num_classes = segmentations.unique().max() + 1
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, segmentation in zip(frames, segmentations):
        img = overlay_segmentation(frame, segmentation, colors, num_classes, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs



def overlay_segmentation(img, segmentation, colors, num_classes, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)

    # trying to always make the background of the 'seashell' color
    background_id = segmentation.sum(dim=(-1, -2)).argmax().item()
    cur_colors = colors[1:].copy()
    cur_colors.insert(background_id, "seashell")

    img_with_seg = draw_segmentation_masks(
            img,
            masks=segmentation.to(torch.bool),
            alpha=alpha,
            colors=cur_colors
        )
    return img_with_seg / 255



def process_objs_masks_dinosaur(frames, masks, H, W):
    """ 
    Computing object representations given frames and masks from 
    an extended-DINOSAUR mode
    """
    B, num_frames, num_objs, _, h, w = masks.shape
    masks = torch.nn.functional.interpolate(
            masks.reshape(B * num_frames * num_objs, 1, h, w),
            size=(H, W),
            mode='nearest'
        ).reshape(B, num_frames, num_objs, 1, H, W)

    frames_resized = torch.nn.functional.interpolate(
            frames[:, :num_frames].reshape(B * num_frames, *frames.shape[-3:]),
            size=(H, W),
            mode='bilinear'
        ).reshape(B, num_frames, 3, H, W)

    objs = frames_resized.unsqueeze(2) * masks
    return objs, masks, frames_resized


#
