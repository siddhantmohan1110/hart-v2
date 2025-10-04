import json

def load_mjhq(path: str):
    """
    path (str): the dataset path

    """


    with open(path, 'r') as f:
        meta_data = json.load(f)

    processed_data = []
    for id, value in meta_data.items():
        processed_data.append({"id": id, "prompt": value['prompt']})

    prompts = [p['prompt'] for p in processed_data]
    return prompts