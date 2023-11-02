# EvaDB LoRA Integration

## The Problem
How does one maintain consistent context between images generated using models such as DALLE or Stable Diffusion? For example, how does one maintain the same artistic style, the same facial structure of characters, or just the same general appearance of characters? This is useful for applications that utilize AI-generated images including creating stories, poems, plays, or artwork collections.

## The Solution

LoRA (Low-rank adaptation) allows for efficient fine-tuning of diffusion models. It allows for extremely small model file sizes of sometimes only a few megabytes. LoRA breaks the weights of the matrices of the cross-attention layer of the model into two smaller, lower-rank matrices. This allows for a much smaller file size. It effectively lowers the number of parameters that are trained. LoRA models can be trained in only a few minutes (around 6), and are extremely good at maintaining consistent style among images between generations.

### Setup and Use Guide
Make sure you have the necessary Python packages installed which can be accomplished via `pip install -r requirements.txt`.

Next, set your replicate API token and training images folder location. `os.environ['REPLICATE_API_TOKEN'] = key` and `location=<path-to-folder>`.

Set your EvaDB cursor with `cursor = evadb.connect().cursor()`
and run `cursor.query("""DROP TABLE IF EXISTS MyImages""").execute()`.

Next Load your images into a new table called MyImages or whatever you choose. 
```
import PIL.Image
for file_number, file_entry in enumerate(os.scandir(location)):
    # filter out non-images first
    print(file_entry)
    print(file_entry.path)
    cursor.query(f"""LOAD IMAGE '{file_entry.path}' INTO MyImages;""").execute()
```

Next run a select query with the StableDiffusionLoRA function, and make sure to provide the number of images in your training data folder. 
```
table = cursor.query("SELECT StableDiffusionLoRA(*) FROM MyImages GROUP BY 'name <num-training-images>';").df()
```

Adjust the generative prompt in the stable_diffusion_lora.py file as necessary, to make it generate the type of image you want.
