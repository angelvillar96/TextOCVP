"""
CATER Dataset

Code adapted from:
  - https://github.com/lkhphuc/simone/blob/main/data/cater.py
  - https://github.com/Youncy-Hu/MAGE/blob/main/dataload.py
"""

import os
import json
import numpy as np
from decord import VideoReader
import torch
import torchvision.transforms.functional as T
from torch.utils.data import Dataset

from lib.logger import print_



class CATER(Dataset):
    """ 
    Class for the CATER Dataset. Multiple CLEVR-like objects moving,
    and text sequences describing the actions taking place.

    Args:
    -----
    root: string
        Path to where the data is stored
    mode: string
        "easy" or "hard"
    tokenizer: string
        Tokenizer used to compute the text indices. Options are "custom" or "T5".
    split: string
        Dataset split to load
    num_frames: int
        Desired length of the sequences to load.
    img_size: int
        Images are resized to this resolution.
    random_start: bool
        If True, first frame of the sequence is sampled at random between the
        possible starting frames.
        Otherwise, starting frame is always the first frame in the sequence.
    """

    TOKENIZERS = ["CustomTokenizer", "T5"]
    MODES = ["easy", "hard"]
    EASY_VOCAB = {
        '[PAD]': 0, '[CLS]': 1, '[SEP]': 2, 'the': 3, 'cone': 4, 'snitch': 5, 'is': 6,
        'sliding': 7, 'picked': 8, 'placed': 9, 'containing': 10, 'rotating': 11,
        'and': 12, 'to': 13, 'up': 14, '(': 15, ')': 16, '1': 17, '2': 18, '3': 19,
        '-1': 20, '-2': 21, '-3': 22, ',': 23, '.': 24, 'first': 25, 'second': 26,
        'third': 27, 'fourth': 28, 'quadrant': 29
    }
    HARD_VOCAB = {
        '[PAD]': 0, '[CLS]': 1, '[SEP]': 2, 'the': 3, 'cone': 4, 'snitch': 5, 'is': 6,
        'sliding': 7, 'picked': 8, 'placed': 9, 'containing': 10, 'and': 11, 'to': 12,
        'up': 13, 'sphere': 14, 'cylinder': 15, 'cube': 16, 'small': 17, 'medium': 18,
        'large': 19, 'metal': 20, 'rubber': 21, 'gold': 22, 'gray': 23, 'red': 24,
        'blue': 25, 'green': 26, 'brown': 27, 'purple': 28, 'cyan': 29, 'yellow': 30,
        '(': 31, ')': 32, '1': 33, '2': 34, '3': 35, '-1': 36, '-2': 37, '-3': 38,
        ',': 39, '.': 40, 'rotating': 41, 'while': 42, 'contained': 43, 'still': 44, 
        'first': 45, 'second': 46, 'third': 47, 'fourth': 48, 'quadrant': 49
    }

    def __init__(self, root, mode, split, tokenizer, img_size=64, num_frames=16,
                 random_start=False, **kwargs):
        """
        Dataset Initializer
        """
        if not os.path.exists(root):
            raise FileNotFoundError(f"{root} does not exist...")
        if tokenizer not in CATER.TOKENIZERS:
            raise NameError(f"{tokenizer = } unknown. Use one of {CATER.TOKENIZERS}...")
        if mode not in CATER.MODES:
            raise NameError(f"{mode = } unknown. Use one of {CATER.MODES}...")
        if split not in ["train", "val", "valid", "test", "eval"]:
            raise ValueError(f"Unknown {split = }...")
        split = "test" if split in ["valid", "test", "eval"] else split

        self.mode = mode
        self.root = os.path.join(root, mode)
        assert os.path.exists(root), f"{self.root} does not exist..."
        self.split = split
        self.tokenizer_name = tokenizer
        self.img_size = img_size
        self.num_frames = num_frames
        self.random_start = random_start
        
        # instanciating tokenizer
        self._setup_tokenizer()
        
        # loading annotations
        with open(os.path.join(self.root, f'{self.split}_explicit.json'), 'r') as f:
            self.annotations = json.load(f)

        print_("Instanciating CATER dataset:")
        print_(f"  --> root: {self.root}")
        print_(f"  --> split: {self.split}")
        print_(f"  --> NumFrames: {self.num_frames}")
        print_(f"  --> ImgSize: {self.img_size}")
        print_(f"  --> RandomStart: {self.random_start}")
        return

    def _setup_tokenizer(self):
        """ Instanciating Tokenizer """
        if self.tokenizer_name == "CustomTokenizer":
            from models.EncodersDecoders.text_encoders import CustomTokenizer
            import nltk
            nltk.download('punkt_tab')
            vocabulary = CATER.EASY_VOCAB if self.mode == "easy" else CATER.HARD_VOCAB
            tokenizer = CustomTokenizer(vocabulary=vocabulary)
        elif self.tokenizer_name == "T5":
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
        else:
            raise NameError(f"Upsi! Unknown tokenizer {self.tokenizer_name}...")
        self.tokenizer = tokenizer
        return

    def __len__(self):
        """ Number of sequences in the dataset """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Sampling a video sequence from the datastet, along with the corresponding text
        """
        # loading video sequence and caption
        video_path = self.annotations[str(idx)]['video']
        video_path = os.path.join(self.root, video_path)
        vid = VideoReader(video_path)
        num_frames = len(vid) #31
        caption = self.annotations[str(idx)]['caption']

        # selecting fixed or random start of a sequence, and loading frames
        if self.random_start and self.split == "train":
            start_frame = np.random.randint(0, num_frames - self.num_frames + 1)
        else:
            start_frame = 1
        choice_idx = np.arange(start_frame, start_frame + self.num_frames)
        images = vid.get_batch(list(choice_idx.astype(np.int32))).asnumpy()

        # frame postprocessing
        images = images[:self.num_frames]
        images = torch.from_numpy(images / 255).permute(0, 3, 1, 2)
        images = T.resize(images, self.img_size).float()
        return images, caption

    def tokenize_captions(self, caption):
        """ Tokenizing a batch of captions"""
        assert isinstance(caption, list)        
        # T5 encoding
        if self.tokenizer_name == "T5":
            t5_out = self.tokenizer(
                    caption,
                    padding=True,
                    return_tensors="pt"
                )
            caption_tokens = t5_out["input_ids"]
            attention_mask = t5_out["attention_mask"]
            caption_lengths = torch.tensor([
                    caption_tokens.shape[1] for _ in range(len(caption_tokens))
                ])
        elif self.tokenizer_name == "CustomTokenizer":
            caption_tokens, caption_lengths = self.tokenizer.tokenize_batch(caption)
            attention_mask = None
        return caption_tokens, caption_lengths, attention_mask

    def collate_fn(self, data):
        """
        Custom collate function.
        Tokenize the text prompts/captions using the T5 tokenizer.
        Since captiones need not have the same length, we pad them with a special
        token so as to have a fixed length.
        """
        images = torch.stack([d[0] for d in data], dim=0)
        raw_captions = [d[1] for d in data]
        tokenizer_out = self.tokenize_captions(raw_captions)
        caption_tokens, caption_lengths, attn_masks = tokenizer_out
        caption_info = {
                "caption": raw_captions,
                "caption_tokens": caption_tokens,
                "caption_lengths": caption_lengths,
                "attn_masks": attn_masks,
            }
        return images, caption_info


#