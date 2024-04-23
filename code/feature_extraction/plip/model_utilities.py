# Imports
from typing import List
import numpy as np
from tqdm import tqdm

# PyTorch Imports
import torch
from torch.utils.data import DataLoader

# Hugging Face Imports
from transformers import CLIPModel, CLIPProcessor
from datasets import Dataset



# Class: PLIP
class PLIP:

    # Method: __init__
    def __init__(self, model_name, device, auth_token=None):
        self.device = device
        self.model_name = model_name
        self.model, self.preprocess, self.model_hash = self._load_model(model_name, auth_token=auth_token)
        self.model = self.model.to(self.device)

        return


    # Method: _load_model
    def _load_model(self, name: str, auth_token=None):

        model = CLIPModel.from_pretrained(name, use_auth_token=auth_token)
        preprocessing = CLIPProcessor.from_pretrained(name, use_auth_token=auth_token)
        model_hash = hash(model)

        return model, preprocessing, model_hash


    # Method: encode_images
    def encode_images(self, images, batch_size, num_workers, pin_memory, progress_bar=False):

        # Build dataset
        # dataset = Dataset.from_dict({'image': images})
        dataset = Dataset.from_dict({'pixel_values': images})
        dataset.set_format('torch')
        # dataset.set_transform(transform_fn)

        # Build dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        
        # Create a list for image embeddings
        image_embeddings = []

        # Show progress bar
        if progress_bar:
            pbar = tqdm(total=len(images) // batch_size, position=0)
        
        # Get image embeddings
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                image_embeddings.extend(self.model.get_image_features(**batch).detach().cpu().numpy())
                if progress_bar:
                    pbar.update(1)
            if progress_bar:
                pbar.close()

        return np.stack(image_embeddings)


    # Method: Encode text (i.e., get text embeddings)
    def encode_text(self, text: List[str], batch_size: int, progress_bar=True):

        # Build dataset
        dataset = Dataset.from_dict({'text': text})
        dataset = dataset.map(
            lambda el: self.preprocess(text=el['text'], return_tensors="pt", max_length=77, padding="max_length", truncation=True),
            batched=True,
            remove_columns=['text']
        )
        dataset.set_format('torch')

        # Build dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Create a list for text embeddings
        text_embeddings = []

        # Show progress_bar
        if progress_bar:
            pbar = tqdm(total=len(text) // batch_size, position=0)
        
        # Get text embeddings
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                text_embeddings.extend(self.model.get_text_features(**batch).detach().cpu().numpy())
                if progress_bar:
                    pbar.update(1)
            if progress_bar:
                pbar.close()

        return np.stack(text_embeddings)


    # Method: Compute cosine similarity
    def _cosine_similarity(self, key_vectors: np.ndarray, space_vectors: np.ndarray, normalize=True):
        if normalize:
            key_vectors = key_vectors / np.linalg.norm(key_vectors, ord=2, axis=-1, keepdims=True)
        return np.matmul(key_vectors, space_vectors.T)


    # Method: Get K-Nearest Neighbours
    def _nearest_neighbours(self, k, key_vectors, space_vectors, normalize=True):

        cosine_sim = self._cosine_similarity(np.array(key_vectors), np.array(space_vectors), normalize=normalize)
        nn = cosine_sim.argsort()[:, -k:][:, ::-1]

        return nn


    # Method: Perform zero-shot image classification
    def zero_shot_classification(self, images, text_labels: List[str], batch_size=8, debug=False):

        # Encode text
        text_vectors = self.encode_text(text_labels, batch_size=batch_size)

        # Encode images
        image_vectors = self.encode_images(images, batch_size=batch_size)

        # Compute cosine similarity
        cosine_sim = self._cosine_similarity(image_vectors, text_vectors)
        

        # Print information (if debug is True)
        if debug:
            print(cosine_sim)
        
        
        # Get predictions
        preds = np.argmax(cosine_sim, axis=-1)


        return [text_labels[idx] for idx in preds]


    # Method: Image retrieval from queries
    def retrieval(self, queries: List[str], top_k: int = 10):

        # Encode text and return K-nearest neighbours
        text_vectors = self.encode_text(queries, batch_size=8)

        return self._nearest_neighbours(k=top_k, key_vectors=text_vectors, space_vectors=self.image_vectors)
