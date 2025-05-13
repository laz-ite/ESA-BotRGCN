import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class Twibot20(Dataset):
    def __init__(self, root='', device='cpu', process=True, save=True):
        self.root = root
        self.device = device
        self.process = process
        self.save = save

        if process:
            self.df_data_labeled, self.df_data = self._load_and_merge_data()
        else:
            self.ensure_data_loaded()

    def _load_and_merge_data(self):
        print('Loading labeled and support data...')
        df_train = pd.read_json(os.path.join(self.root, "train.json"))
        df_dev = pd.read_json(os.path.join(self.root, "dev.json"))
        df_test = pd.read_json(os.path.join(self.root, "test.json"))
        df_support = pd.read_json(os.path.join(self.root, "support.json"))

        df_labeled = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df_support = df_support.iloc[:, [0, 1, 2, 3]]
        df_support['label'] = 'None'

        df_train = df_train.iloc[:, [0, 1, 2, 3, 5]]
        df_dev = df_dev.iloc[:, [0, 1, 2, 3, 5]]
        df_test = df_test.iloc[:, [0, 1, 2, 3, 5]]

        df_all = pd.concat([df_train, df_dev, df_test, df_support], ignore_index=True)
        print('Finished loading data.')

        return df_labeled, df_all

    def ensure_data_loaded(self):
        print("Manually loading data...")
        base_path = os.path.join(self.root, "Twibot-20")
        files = ["train.json", "dev.json", "test.json", "support.json"]
        dataframes = [pd.read_json(os.path.join(base_path, f)) for f in files]

        self.df_data_labeled = pd.concat(dataframes[:3], ignore_index=True)
        self.df_data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.df_data)} records.")

    def load_labels(self):
        print('Loading labels...', end=' ')
        path = os.path.join(self.root, 'Data', 'label.pt')
        if not os.path.exists(path):
            labels = torch.LongTensor(self.df_data_labeled['label'].values).to(self.device)
            if self.save:
                torch.save(labels, path)
        else:
            labels = torch.load(path).to(self.device)
        print('Done.')
        return labels

    def dataloader(self):
        self.ensure_data_loaded()
        des_tensor = self.Des_embbeding()
        tweets_tensor = self.tweets_embedding()
        num_prop = self.num_prop_preprocess()
        category_prop = self.cat_prop_preprocess()
        edge_index, edge_type = self.Build_Graph()
        labels = self.load_labels()
        new_feature = self.load_new_feature()
        train_idx, val_idx, test_idx = self.train_val_test_mask()

        return (des_tensor, tweets_tensor, num_prop, category_prop,
                new_feature, edge_index, edge_type, labels,
                train_idx, val_idx, test_idx)

    def Des_Preprocess(self):
        print('Loading descriptions...', end=' ')
        path = os.path.join(self.root, 'Data', 'description.npy')
        if not os.path.exists(path):
            description = [
                (p.get('description') if p and p.get('description') else 'None')
                for p in self.df_data['profile']
            ]
            description = np.array(description)
            if self.save:
                np.save(path, description)
        else:
            description = np.load(path, allow_pickle=True)
        print('Done.')
        return description

    def load_new_feature(self):
        print("Loading new feature...", end=' ')
        path = os.path.join(self.root, "Data", "max_emotion_span_tensor_smile.pt")
        if not os.path.exists(path):
            new_feature = self.generate_new_feature()
            if self.save:
                torch.save(new_feature, path)
        else:
            new_feature = torch.load(path).to(self.device)
        print("Done.")
        return new_feature

    def generate_new_feature(self):
        print("Generating new feature...")
        return torch.randn(self.df_data.shape[0], 128)

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import pipeline

class Twibot20Embeddings:
    def __init__(self, root, device='cpu', save=True):
        self.root = root
        self.device = device
        self.save = save

    def _load_or_generate_tensor(self, path, generate_fn):
        if not os.path.exists(path):
            tensor = generate_fn()
            if self.save:
                torch.save(tensor, path)
        else:
            tensor = torch.load(path).to(self.device)
        return tensor

    def des_embedding(self):
        print('Running description embedding...')
        path = os.path.join(self.root, "Data", "des_tensor.pt")

        def generate_des():
            desc_path = os.path.join(self.root, "Data", "description.npy")
            descriptions = np.load(desc_path, allow_pickle=True)
            extractor = pipeline('feature-extraction', model="distilroberta-base", tokenizer="distilroberta-base", device=0)

            embeddings = []
            for desc in tqdm(descriptions):
                features = torch.tensor(extractor(desc)[0])
                pooled = features.mean(dim=0)
                embeddings.append(pooled)

            return torch.stack(embeddings).to(self.device)

        return self._load_or_generate_tensor(path, generate_des)

    def tweets_embedding(self):
        print('Running tweets embedding...')
        path = os.path.join(self.root, "Data", "tweets_tensor.pt")

        def generate_tweets():
            tweets_path = os.path.join(self.root, "Data", "tweets.npy")
            tweets = np.load(tweets_path, allow_pickle=True)
            extractor = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=0,
                                 padding=True, truncation=True, max_length=500)

            embeddings = []
            for tweet_list in tqdm(tweets):
                tweet_vecs = []
                for tweet in tweet_list:
                    features = torch.tensor(extractor(tweet)[0])
                    tweet_vecs.append(features.mean(dim=0))
                user_vec = torch.stack(tweet_vecs).mean(dim=0)
                embeddings.append(user_vec)

            return torch.stack(embeddings).to(self.device)

        return self._load_or_generate_tensor(path, generate_tweets)

    def load_sentiment(self):
        print("Loading sentiment feature...", end=' ')
        path = os.path.join(self.root, "Data", "sentiment_tensor.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing sentiment feature: {path}")

        sentiment_tensor = torch.load(path).to(self.device)
        if sentiment_tensor.dim() == 1:
            sentiment_tensor = sentiment_tensor.unsqueeze(1)

        print("Done.")
        return sentiment_tensor

    def tweets_preprocess(self, df_data):
        print("Loading raw tweets...", end=' ')
        path = os.path.join(self.root, "Data", "tweets.npy")
        if not os.path.exists(path):
            tweets = [
                [tweet if tweet else '' for tweet in (row['tweet'] or [''])]
                for _, row in df_data.iterrows()
            ]
            np.save(path, np.array(tweets, dtype=object))
        else:
            tweets = np.load(path, allow_pickle=True)
        print("Done.")
        return tweets


def num_prop_preprocess(self):
    print('Processing numerical features...', end='   ')
    path = os.path.join(self.root, "Data", "num_properties_tensor.pt")

    if not os.path.exists(path):
        base_path = os.path.join(self.root, "Data")
        fields = [
            "active_days", "screen_name_length", "favourites_count",
            "followers_count", "friends_count", "statuses_count", "sentiment_tensor"
        ]

        def load_and_normalize(fname):
            tensor = torch.load(os.path.join(base_path, f"{fname}.pt")).to(self.device)
            if tensor.dim() == 1:
                tensor = tensor.reshape(-1, 1)
            arr = tensor.cpu().numpy()
            return torch.tensor((arr - arr.mean()) / arr.std(), dtype=torch.float32).to(self.device)

        features = [load_and_normalize(field) for field in fields]
        num_prop = torch.cat(features, dim=1)

        if self.save:
            torch.save(num_prop, path)
    else:
        num_prop = torch.load(path).to(self.device)

    print("Finished")
    return num_prop


def cat_prop_preprocess(self):
    print('Processing categorical features...', end='   ')
    path = os.path.join(self.root, 'Data', 'cat_properties_tensor.pt')

    if not os.path.exists(path):
        binary_fields = [
            'protected', 'geo_enabled', 'verified', 'contributors_enabled', 'is_translator',
            'is_translation_enabled', 'profile_background_tile', 'profile_use_background_image',
            'has_extended_profile', 'default_profile', 'default_profile_image'
        ]

        def extract_props(profile):
            if not profile:
                return [0] * len(binary_fields)
            return [1 if profile.get(field) == "True " else 0 for field in binary_fields]

        category_props = [extract_props(p) for p in self.df_data['profile']]
        category_tensor = torch.tensor(category_props, dtype=torch.float32).to(self.device)

        if self.save:
            torch.save(category_tensor, path)
    else:
        category_tensor = torch.load(path).to(self.device)

    print('Finished')
    return category_tensor


def Build_Graph(self):
    print('Building graph...', end='   ')
    edge_path = os.path.join(self.root, 'Data', 'edge_index.pt')
    type_path = os.path.join(self.root, 'Data', 'edge_type.pt')

    if not os.path.exists(edge_path):
        id2idx = {id_: idx for idx, id_ in enumerate(self.df_data['ID'])}
        edges, types = [], []

        for idx, neighbors in enumerate(self.df_data['neighbor']):
            if neighbors:
                for rel_type, key in enumerate(['following', 'follower']):
                    for neighbor_id in neighbors.get(key, []):
                        target = id2idx.get(int(neighbor_id))
                        if target is not None:
                            edges.append([idx, target])
                            types.append(rel_type)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_type = torch.tensor(types, dtype=torch.long).to(self.device)

        if self.save:
            torch.save(edge_index, edge_path)
            torch.save(edge_type, type_path)
    else:
        edge_index = torch.load(edge_path).to(self.device)
        edge_type = torch.load(type_path).to(self.device)

    print('Finished')
    return edge_index, edge_type


def train_val_test_mask(self):
    return range(8278), range(8278, 10643), range(10643, 11826)












