""" 
Module for loading the CLIPort dataset
"""

import os
import random
import torch
from torchvision import transforms
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from lib.logger import print_
from PIL import Image



class CLIPort:
    """
    Module for loading data from the CLIPort dataset.
    A robot arm must pick-up a block and place it in a bowl.

    Args:
    -----
    root: str
        Root directory of the dataset.
    tokenizer: string
        Tokenizer used to compute the text indices. Options are "custom" or "T5".
    split: string
        Dataset split to load
    num_frames: int, optional
        Number of frames to load for each episode. Default: None
    img_size: tuple, optional
        Size to which frames should be resized.
        If None, keep original size. Default: None
    random_start: bool, optional
        Whether to select a random start frame during training. Default: False
    """
    
    VOCABULARY = {
            '[PAD]': 0, 
            '[CLS]': 1, 
            '[SEP]': 2,
            'block': 3,
            'blue': 4,
            'bowl': 5,
            'brown': 6,
            'cyan': 7,
            'gray': 8,
            'green': 9,
            'in': 10,
            'put': 11,
            'red': 12,
            'the': 13,
            'yellow': 14
        }

    # vocabulary for test-set with unseen colors
    VOCABULARY_TEST = {
            '[PAD]': 0, 
            '[CLS]': 1, 
            '[SEP]': 2,
            'block': 3,
            'blue': 4,
            'bowl': 5,
            'pink': 6,
            'purple': 7,
            'orange': 8,
            'green': 9,
            'in': 10,
            'put': 11,
            'red': 12,
            'the': 13,
            'white': 14
        }
    EXCLUDE_EPISODES = ["episode07564", "episode09031", "episode13755", "episode11237"]
    TOKENIZERS = ["CustomTokenizer", "T5"]


    def __init__(self, root, split, tokenizer, num_frames, img_size,
                 random_start=False, **kwargs):
        """
        Dataset initializer
        """
        if not os.path.exists(root):
            raise FileNotFoundError(f"{root} does not exist...")
        if tokenizer not in CLIPort.TOKENIZERS:
            raise NameError(f"{tokenizer = } unknown. Use one of {CLIPort.TOKENIZERS}...")
        if split not in ["train", "val", "valid", "test", "eval"]:
            raise ValueError(f"Unknown {split = }...")
        split = "val" if split in ["val", "valid"] else split
        split = "test" if split in ["eval", "test"] else split

        self.root = os.path.join(f"{root}", f"{split}")
        self.split = split
        self.tokenizer_name = tokenizer
        self.num_frames = num_frames
        self.img_size = img_size
        self.random_start = random_start if split == "train" else False

        # utils for resizing the images
        self.image_transform = transforms.Compose([
                transforms.Resize(
                        self.img_size,
                        interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                transforms.ToTensor()
            ])
                
        # instanciating tokenizer
        self._setup_tokenizer()

        # Finding available episodes and loading textual captions
        self.episodes = self.fetch_episodes()
        self.num_episodes = len(self.episodes)
        with ThreadPoolExecutor() as executor:
            self.labels = list(tqdm(
                    executor.map(self.load_label, self.episodes),
                    total=self.num_episodes
                ))            
            
        # logging
        print_(f"Loaded CLIPort PutBlockInBowl")
        print_(f"  --> split: {self.split}")
        print_(f"  --> num_frames: {self.num_frames}")
        print_(f"  --> episodes: {self.num_episodes}")
        print_(f"  --> img_size: {self.img_size}")
        print_(f"  --> random_start: {self.random_start}")
        return
    
    def _setup_tokenizer(self):
        """ Instanciating Tokenizer """
        if self.tokenizer_name == "CustomTokenizer":
            from models.EncodersDecoders.text_encoders import CustomTokenizer
            if self.split == "test":
                vocabulary = CLIPort.VOCABULARY_TEST
            else:
                vocabulary = CLIPort.VOCABULARY
            tokenizer = CustomTokenizer(vocabulary=vocabulary)
        elif self.tokenizer_name == "T5":
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
        else:
            raise NameError(f"Upsi! Unknown tokenizer {self.tokenizer_name}...")
        self.tokenizer = tokenizer
        return
        

    def __len__(self):
        """ Number of episodes in the dataset """
        return self.num_episodes


    def __getitem__(self, idx):
        """ Get an episode from the dataset """
        cur_episode = self.episodes[idx]
        caption = self.labels[idx]
        color_frames, start_frame_idx = self.load_episode(cur_episode)
        metas = {
            "episode": cur_episode,
            "start_frame_idx": start_frame_idx
        }
        return color_frames, caption, metas    


    def fetch_episodes(self):
        """ Fetching valid episodes """
        all_episodes = [
                f for f in os.listdir(self.root)
                if f.startswith("episode") and f not in CLIPort.EXCLUDE_EPISODES
            ]
        all_episodes = sorted(all_episodes, key=lambda x: int(x.split("episode")[-1]))        
        return all_episodes


    def load_label(self, episode_dir):
        """
        Reading all task descriptions
        """
        task_caption_file = os.path.join(self.root, episode_dir, 'task_description.txt')
        if not os.path.exists(task_caption_file):
            raise FileNotFoundError(f"Task-caption file not found: {task_caption_file}")
        with open(task_caption_file, 'r') as f:
            label = f.read().strip()
        return label


    def load_episode(self, episode):
        """
        Loading imagess for the corresponding episode
        """
        color_dir = os.path.join(self.root, episode, 'color')
        assert os.path.exists(color_dir), f"RBG Img-Dir does not exist for {episode}"

        frame_files = sorted(os.listdir(color_dir))
        num_frames = len(frame_files)
        if num_frames < (self.num_frames) :
            raise ValueError(f"{self.num_frames = } are required, but only " +
                             f"{num_frames = } are available for {episode = }...")

        # sequence start and end
        if self.random_start:
            max_start = max(num_frames - (self.num_frames), 0)
            start_frame_idx = random.randint(0, max_start)
        else:
            start_frame_idx = 0
        frame_indices = range(start_frame_idx, start_frame_idx + self.num_frames)

        # loading images
        color_frames = []
        for idx in frame_indices:
            frame_file = frame_files[idx]
            frame_num = frame_file.split("_")[0]
            color_frame = self._load_img(
                    os.path.join(color_dir, f"{frame_num}_color.png")
                )
            color_frames.append(color_frame)
        color_frames = torch.stack(color_frames)
        return color_frames, start_frame_idx
    
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
            caption_lengths = caption_tokens.shape[1]
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
                "attn_masks": attn_masks
            }
        return images, caption_info


    def _load_img(self, p):
        """ Loading image and converting to tensor"""
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        return self.image_transform(img)
