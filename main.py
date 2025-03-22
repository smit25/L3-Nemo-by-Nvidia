import os
import json
import pandas as pd
import numpy as np
import torch

# NeMo imports
import nemo
import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

class PersonalizedContentGenerator:
    def __init__(self, pretrained_model_name = "nemo/gpt2-1.3B", model_path = None,):
      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            if model_path and os.path.exists(model_path):
                print(f"Loading fine-tuned model from {model_path}")
                self.model = MegatronGPTModel.restore_from(model_path)
            else:
                print(f"Loading pre-trained model: {pretrained_model_name}")
                self.model = MegatronGPTModel.from_pretrained(pretrained_model_name)
            
            self.model = self.model.to(self.device)
            self.model.eval() 
        except Exception as e:
            print(f"Error loading model: {e}")
        

    def load_user_data(self, data_path, size_limit=5):
        try:
            df = pd.read_csv(data_path)
            return df.head(size_limit) if size_limit else df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame(columns=['user_id', 'content_name', 'rating', 'watch_time_minutes'])


    def fetch_content_metadata(self, content_name):
        
        metadata = self._generate_mock_metadata(content_name)
        
        return metadata


    def _generate_mock_metadata(self, content_name):
        name_hash = sum(ord(c) for c in content_name) % 100
        
        genres = ["Action", "Comedy", "Drama", "Science Fiction", "Thriller", 
                 "Romance", "Horror", "Adventure", "Fantasy", "Animation"]
        
        keywords = ["suspenseful", "funny", "emotional", "thought-provoking",
                   "violent", "inspirational", "dark", "uplifting", "romantic",
                   "scary", "family-friendly", "visually-stunning"]
        
        selected_genres = [genres[i % len(genres)] for i in range(name_hash % 3 + 1)]
        selected_keywords = [keywords[(name_hash + i) % len(keywords)] for i in range(3)]
        
        return {
            "title": content_name,
            "year": f"20{(name_hash % 23):02d}",
            "genres": selected_genres,
            "keywords": selected_keywords,
        }

    def create_user_profile(self, user_id, user_data):
        user_movies = user_data[user_data['user_id'] == user_id]
        
        if user_movies.empty:
            return {"user_id": user_id, "profile_type": "new_user"}
        
        avg_rating = user_movies['rating'].mean()
        avg_watch_time = user_movies['watch_time_minutes'].mean()
        total_watched = len(user_movies)
        
        genre_counts = {}
        keyword_counts = {}
        
        for _, row in user_movies.iterrows():
            metadata = self.fetch_content_metadata(row['content_name'])
            
            for genre in metadata.get('genres', []):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                
            for keyword in metadata.get('keywords', []):
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_genres = [g[0] for g in top_genres] if top_genres else []
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_keywords = [k[0] for k in top_keywords] if top_keywords else []
        
        if avg_rating > 4.0 and total_watched > 10:
            profile_type = "enthusiast"
        elif avg_rating > 3.5:
            profile_type = "regular_viewer"
        elif total_watched > 20:
            profile_type = "high_volume"
        else:
            profile_type = "casual_viewer"
            
        profile = {
            "user_id": user_id,
            "profile_type": profile_type,
            "favorite_genres": top_genres,
            "favorite_keywords": top_keywords,
            "avg_rating": float(avg_rating) if not np.isnan(avg_rating) else 0.0,
            "avg_watch_time": float(avg_watch_time) if not np.isnan(avg_watch_time) else 0.0,
            "total_watched": total_watched
        }
        
        return profile

    def get_user_profile(self, user_id, user_data):
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        return self.create_user_profile(user_id, user_data)


    def create_prompt(self, content_metadata, user_profile):
        prompt = f"Title: {content_metadata.get('title', '')}\n"
        prompt += f"Year: {content_metadata.get('year', '')}\n"
        prompt += f"Genre: {', '.join(content_metadata.get('genres', []))}\n"
        prompt += f"Director: {', '.join(content_metadata.get('directors', []))}\n"
        prompt += f"Cast: {', '.join(content_metadata.get('cast', [])[:3])}\n\n"
        
        prompt += "User preferences:\n"
        prompt += f"Favorite genres: {', '.join(user_profile.get('favorite_genres', []))}\n"
        prompt += f"Favorite elements: {', '.join(user_profile.get('favorite_keywords', []))}\n"
        prompt += f"Viewer type: {user_profile.get('profile_type', 'general')}\n\n"
        
        prompt += "Create a personalized content description that emphasizes aspects most relevant to this user's preferences:\n"
        
        return prompt

    def generate_description(self, content_name, user_id, user_data):

        content_metadata = self.fetch_content_metadata(content_name)
        
        user_profile = self.get_user_profile(user_id, user_data)
        
        prompt = self.create_prompt(content_metadata, user_profile)
        
        if self.model is not None:
            try:
                outputs = self.model.generate(
                    [prompt],
                    max_length=200,
                    min_length=50,
                    temperature=0.5,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    num_return_sequences=1
                )
                
                generated_text = outputs[0]
                if generated_text.startswith(prompt):
                    description = generated_text[len(prompt):].strip()
                else:
                    description = generated_text.strip()
                
                if "<|endoftext|>" in description:
                    description = description.split("<|endoftext|>")[0].strip()
            except Exception as e:
                print(f"Error generating with model: {e}")
        
        return {
            "content_name": content_name,
            "user_id": user_id,
            "user_profile_type": user_profile.get("profile_type", "general"),
            "description": description,
            "metadata": content_metadata
        }

  
    def batch_generate(self,  user_data_path, output_path,limit = None):
     
        user_data = self.load_user_data(user_data_path)
        
        pairs = user_data[['user_id', 'content_name']].drop_duplicates()
        if limit:
            pairs = pairs.head(limit)
        
        results = []
        
        for idx, (_, row) in enumerate(pairs.iterrows()):
            user_id = row['user_id']
            content_name = row['content_name']
            
            print(f"Generating description {idx+1}/{len(pairs)}: {content_name} for user {user_id}")
            
            description = self.generate_description(content_name, user_id, user_data)
            results.append(description)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Generated {len(results)} personalized descriptions, saved to {output_path}")


    def finetune_model(self, train_data_path, output_model_path,epochs= 3,batch_size= 4):
        if self.model is None:
            print("Error: No model loaded for fine-tuning")
            return
            
        try:
            print(f"Fine-tuning model, this may take a while...")
            
            self.model.cfg.precision = 16  # Use mp
            self.model.cfg.optim.lr = 5e-5
            self.model.cfg.optim.weight_decay = 0.01
            
            train_data_config = {
                'file_path': train_data_path,
                'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 2
            }
            
            self.model.setup_training_data(train_data_config)
            
            trainer = nemo.core.Trainer(
                gpus=1 if torch.cuda.is_available() else 0,
                max_epochs=epochs,
                precision=16 if torch.cuda.is_available() else 32,
                accumulate_grad_batches=4, 
            )
            
            trainer.fit(self.model)
            
            save_path = self.model.save_to(output_model_path)
            print(f"Model fine-tuned and saved to {save_path}")
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")


def main():

    
    model_path = None # fine-tuned model
    generator = PersonalizedContentGenerator(model_path=model_path)
    
    
    user_data_path = "sample_data.csv" 
    output_path = "generated_descriptions.json"   
    limit = 5
    generator.batch_generate(user_data_path, output_path, limit)
    
    # Generate single description
    # content_name = "Inception"
    # user_id = "user123"
    # user_data_path = "sample_data.csv"
    # user_data = generator.load_user_data(user_data_path)
    # result = generator.generate_description(content_name, user_id, user_data)


if __name__ == "__main__":
    main()
