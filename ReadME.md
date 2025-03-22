# Personalized Content Description Generator

A minimal, single-file implementation that uses NVIDIA NeMo to generate personalized content descriptions and summaries. This tool is designed to work with limited data - only requiring content names, user ratings, and watch times to generate customized descriptions that align with user preferences.

## Features

- Works with minimal user data (just content names, ratings, watch times)
- Automatically infers user preferences from viewing patterns
- Generates personalized content descriptions tailored to user interests
- Supports fine-tuning on custom examples
- Simple, self-contained implementation in a single file

## Requirements

- Python 3.8+
- PyTorch 1.13+
- NVIDIA GPU with CUDA support (recommended for faster inference and required for fine-tuning)
- NVIDIA NeMo framework

## Installation

1. Install PyTorch for your CUDA version:
   ```
   pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Install NeMo and dependencies:
   ```
   pip install nemo_toolkit[nlp]
   ```

3. Download the script and make it executable:
   ```
   chmod +x personalized_description_generator.py
   ```

## Quick Start

### Generate a Single Description

```bash
python personalized_description_generator.py describe \
  --content "Inception" \
  --user "user123" \
  --user_data viewing_history.csv
```

### Batch Generate Descriptions

```bash
python personalized_description_generator.py generate \
  --user_data viewing_history.csv \
  --output descriptions.json \
  --limit 10
```

### Fine-tune the Model (requires GPU)

```bash
python personalized_description_generator.py finetune \
  --train_data training_examples.jsonl \
  --output_model my_finetuned_model.nemo \
  --epochs 3
```

## Input Data Format

The script expects a CSV file with user viewing data in the following format:

```
user_id,content_name,rating,watch_time_minutes
user123,Inception,4.5,120
user123,The Matrix,5.0,138
user456,Interstellar,3.5,95
```

## How It Works

1. **User Profile Creation**: 
   - Analyzes viewing history (content watched, ratings, watch times)
   - Identifies genre preferences and viewing patterns
   - Categorizes users into profile types (enthusiast, casual_viewer, etc.)

2. **Content Metadata Enrichment**:
   - Fetches or generates detailed content metadata
   - Extracts relevant content attributes (genre, plot elements, etc.)

3. **Personalized Description Generation**:
   - Creates prompts combining content info and user preferences
   - Uses NeMo's language model to generate tailored descriptions
   - Emphasizes content aspects most relevant to the user's interests

4. **Model Fine-tuning** (optional):
   - Uses examples of personalized descriptions
   - Adapts the language model to better match your specific use case

## Example Usage Scenarios

### Streaming Platform

Enhance content discovery by showing users personalized descriptions that highlight aspects of movies/shows they care about:

- For horror fans: emphasis on atmosphere and scare factors
- For character-driven viewers: focus on character development and relationships
- For action enthusiasts: highlight exciting sequences and pacing

### Content Recommendation

Improve recommendation systems by generating summaries that explain *why* a particular user might enjoy recommended content based on their specific interests and viewing patterns.

## Training Your Own Model

To train a custom model, you'll need to create a training dataset in JSONL format with prompt-completion pairs:

```jsonl
{"prompt": "Title: Inception\nYear: 2010\nGenre: Science Fiction, Action\nDirector: Christopher Nolan\nCast: Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page\n\nUser preferences:\nFavorite genres: Science Fiction, Mystery\nFavorite elements: mind-bending, plot-twists, philosophical\nViewer type: enthusiast\n\nCreate a personalized content description that emphasizes aspects most relevant to this user's preferences:\n", "completion": "Inception (2010) explores mind-bending concepts of shared consciousness through revolutionary dream-tech that allows skilled extractors to navigate layered dreamscapes. Director Christopher Nolan crafts a meticulously designed world where time dilates across dream levels and reality blurs, following Dom Cobb's team as they attempt the supposedly impossible task of inception—planting an idea rather than stealing one. The film's intricate rules of dream architecture and stunning visual representations of collapsing physical laws deliver a cerebral sci-fi experience that questions the nature of reality itself."}
```



## Acknowledgments

- NVIDIA NeMo team for their excellent conversational AI framework
- This implementation is designed for educational purposes and as a starting point for more comprehensive solutions